"""Measured, loss-preserving preview delivery. No model/GPU changes.

Reference/candidate differ only in MP4 fragment duration (4/1 frames).
Codec, GOP, CRF, pixel format, resolution and RGB frame sequence are identical.
Host observations of FFmpeg stderr are not exclusive GPU/codec kernel times.
"""
import collections,json,os,re,socket,struct,subprocess,threading,time
from pathlib import Path
from .records import StreamRecord, PREVIEW_FORMATS, SOURCE_FPS

MODE='frame'
def now():return time.monotonic_ns()/1e6

def boxes(data):
    pos=0
    while pos<len(data):
        if len(data)-pos<8:raise ValueError('truncated MP4 box')
        size,kind=struct.unpack_from('>I4s',data,pos)
        if size<8 or pos+size>len(data):raise ValueError('invalid MP4 box size')
        yield kind,data[pos+8:pos+size]
        pos+=size

def sample_count(moof):
    count=0
    for kind,body in boxes(moof[8:]):
        if kind==b'traf':
            for k,b in boxes(body):
                if k==b'trun':count+=struct.unpack_from('>I',b,4)[0]
    if not 1<=count<=128:raise ValueError('unsupported fragment sample count')
    return count

class FragmentedPreviewWriter:
    def __init__(self,output,*,resolution,ffmpeg,sequence,expected_frames=None):
        if resolution not in PREVIEW_FORMATS or type(sequence)!=int or sequence<1:
            raise ValueError('invalid preview')
        if not output.is_absolute() or output.suffix!='.mp4' or output.exists() or output.is_symlink():
            raise ValueError('fresh owned output required')
        self.output=output;self.part=output.with_name('.'+output.stem+'.part.mp4')
        self.stderr_path=output.with_suffix('.ffmpeg.log')
        self.trace_path=output.with_suffix('.delivery.jsonl')
        for p in (self.part,self.stderr_path,self.trace_path):
            if p.exists() or p.is_symlink():raise FileExistsError(p)
        output.parent.mkdir(parents=True,exist_ok=True)
        self.width,self.height,_=PREVIEW_FORMATS[resolution]
        self.frames=0;self.closed=False;self.expected_frames=expected_frames;self.mode=MODE
        self.record=StreamRecord(sequence,self.part,output)
        self.record.delivery=self
        self.condition=threading.Condition();self.trace_lock=threading.Lock()
        self.entries=collections.deque(maxlen=8192);self.serial=0;self.offset=0
        self.inputs={};self.encoded={};self.error=None;self.output_done=False
        self.trace=self.trace_path.open('x',buffering=1)
        self.stderr=self.stderr_path.open('xb')
        self.file=self.part.open('xb',buffering=0)
        duration='62500' if self.mode=='frame' else '250000'
        command=[ffmpeg,'-hide_banner','-debug_ts','-loglevel','verbose','-nostdin',
            '-f','rawvideo','-pix_fmt','rgb24','-s:v',f'{self.width}x{self.height}',
            '-r',str(SOURCE_FPS),'-i','pipe:0','-an','-c:v','libx264',
            '-preset','veryfast','-tune','zerolatency','-crf','24',
            '-g','4','-keyint_min','4','-sc_threshold','0','-pix_fmt','yuv420p',
            '-threads','2','-flush_packets','1',
            '-movflags','frag_keyframe+empty_moov+default_base_moof',
            '-frag_duration',duration,'-f','mp4','pipe:1']
        self.command=command
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,start_new_session=True,bufsize=0)
        self.log('configuration',mode=self.mode,command=command)
        self.reader=threading.Thread(target=self._read,name='preview-fragments',daemon=True)
        self.logger=threading.Thread(target=self._stderr,name='preview-encoder-timing',daemon=True)
        self.reader.start();self.logger.start()

    def log(self,event,**row):
        with self.trace_lock:self.trace.write(json.dumps(dict(event=event,at_ms=now(),**row))+'\n')

    def _stderr(self):
        for line in self.process.stderr:
            stamp=now();text=line.decode(errors='replace')
            # Do not retain debug output for every demux/filter step indefinitely.
            if 'encoder -> type:video' in text:
                match=re.search(r'pkt_pts:(\d+)',text)
                if match:
                    index=int(match[1])
                    with self.condition:self.encoded[index]=stamp
                    self.log('encoded',frame=index,observed_ms=stamp)
            elif 'muxer <- pts:' in text and 'latency(' in text:
                match=re.search(r'pts:(\d+).*latency\(total:([\d.]+)ms',text)
                enc=re.search(r'[, ]encode ([\d.]+)ms',text)
                if match:self.log('ffmpeg_latency',pts=int(match[1]),total_ms=float(match[2]),
                    encode_ms=float(enc[1]) if enc else None)
            elif 'error' in text.lower() or 'failed' in text.lower():self.stderr.write(line)

    def _exact(self,n):
        data=bytearray()
        while len(data)<n:
            part=self.process.stdout.read(n-len(data))
            if not part:
                if not data:return None
                raise RuntimeError('truncated encoder output')
            data.extend(part)
        return bytes(data)

    def _publish(self,data,first,count):
        stamp=now();offset=self.offset
        self.file.write(data);self.offset+=len(data)
        with self.condition:
            info=dict(serial=self.serial,offset=offset,length=len(data),first=first,count=count,
                      fragment_ready_ms=stamp,mode=self.mode)
            self.serial+=1;self.entries.append(info);self.condition.notify_all()
        self.log('fragment',**info)
        if count:
            with self.condition:
                for index in range(first,first+count):
                    self.inputs.pop(index,None);self.encoded.pop(index,None)

    def _read(self):
        pending=bytearray();first=0;count=0
        try:
            while True:
                head=self._exact(8)
                if head is None:break
                size,kind=struct.unpack('>I4s',head)
                if not 8<=size<=32*1024*1024:raise RuntimeError('unsupported output box')
                body=self._exact(size-8) if size>8 else b''
                if body is None:raise RuntimeError('missing box body')
                data=head+body
                if kind==b'moof':
                    if pending:raise RuntimeError('unexpected pending box')
                    count=sample_count(data);pending.extend(data)
                elif kind==b'mdat':
                    if not pending or not count:raise RuntimeError('orphan mdat')
                    pending.extend(data);self._publish(bytes(pending),first,count)
                    first+=count;count=0;pending.clear()
                else:self._publish(data,first,0)
            if pending:raise RuntimeError('incomplete final fragment')
            self.output_frames=first
        except BaseException as e:
            self.error=str(e)
        finally:
            self.file.close()
            with self.condition:self.output_done=True;self.condition.notify_all()

    def append_rgb(self,array,index):
        if self.closed or index!=self.frames or tuple(array.shape)!=(self.height,self.width,3) or str(array.dtype)!='uint8':
            raise RuntimeError('RGB frame order/shape differs')
        if self.expected_frames is not None and index>=self.expected_frames:raise RuntimeError('horizon exceeded')
        if self.error or self.process.poll() is not None:raise RuntimeError('preview encoder stopped: '+str(self.error))
        self.log('rgb_input',frame=index)
        data=memoryview(array).cast('B');written=0
        while written<len(data):
            n=self.process.stdin.write(data[written:])
            if not n:raise RuntimeError('preview pipe closed')
            written+=n
        self.log('rgb_submitted',frame=index);self.frames+=1

    def finish(self):
        if self.closed:raise RuntimeError('preview already closed')
        self.closed=True;self.process.stdin.close()
        code=self.process.wait(timeout=120);self.reader.join(10);self.logger.join(10)
        if self.reader.is_alive() or self.logger.is_alive():raise RuntimeError('encoder observers did not exit')
        self.log('finish',frames=self.frames,output_frames=getattr(self,'output_frames',None),rc=code,error=self.error)
        self.stderr.close();self.trace.close()
        if code or self.error or getattr(self,'output_frames',None)!=self.frames:raise RuntimeError('encoder incomplete')
        if self.expected_frames is not None and self.frames!=self.expected_frames:raise RuntimeError('incomplete horizon')
        os.replace(self.part,self.output);return self.output

    def abort(self):
        self.record.done.set()
        if self.closed:return
        self.closed=True
        if self.process.poll() is None:self.process.terminate()
        try:self.process.wait(10)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(10)
        self.reader.join(10);self.logger.join(10);self.stderr.close();self.trace.close()

def stream(handler,record):
    writer=getattr(record,'delivery',None)
    if writer is None:return handler._json(409,{'error':'measured stream not available'})
    handler.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    handler.send_response(200);handler.send_header('Content-Type','application/x-h3-fragments')
    handler.send_header('Cache-Control','no-store');handler.send_header('Transfer-Encoding','chunked')
    handler.end_headers();cursor=0
    try:
        with writer.part.open('rb') if writer.part.exists() else writer.output.open('rb') as source:
            while True:
                with writer.condition:
                    writer.condition.wait_for(lambda:writer.serial>cursor or writer.output_done,timeout=15)
                    rows=[r for r in writer.entries if r['serial']>=cursor]
                    if rows and rows[0]['serial']!=cursor:raise RuntimeError('client fell behind retained fragment index')
                    done=writer.output_done
                for row in rows:
                    source.seek(row['offset']);data=source.read(row['length'])
                    if len(data)!=row['length']:raise RuntimeError('fragment file incomplete')
                    meta=json.dumps(dict(row,send_ms=now()),separators=(',',':')).encode()
                    packet=struct.pack('>II',len(meta),len(data))+meta+data
                    handler.wfile.write(f'{len(packet):x}\r\n'.encode()+packet+b'\r\n');handler.wfile.flush()
                    cursor=row['serial']+1
                if done:
                    if writer.error:raise RuntimeError(writer.error)
                    handler.wfile.write(b'0\r\n\r\n');handler.wfile.flush();return
    except (OSError,RuntimeError):handler.close_connection=True
