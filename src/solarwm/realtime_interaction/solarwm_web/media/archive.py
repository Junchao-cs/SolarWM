"""Full-resolution archival encoding, separate from CRF24 live preview."""
import subprocess


class Archive:
    def __init__(self, output, ffmpeg):
        self.output = output
        self.part = output.with_name("." + output.stem + ".archive-part.mp4")
        if output.exists() or self.part.exists():
            raise FileExistsError(output)
        self.log = output.with_suffix(".archive.log").open("xb")
        # Matches the former imageio quality=8 / libx264 / yuv420p archive.
        self.process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error",
            "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", "1344x768",
            "-r", "16", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "10",
            "-threads", "8", "-pix_fmt", "yuv420p", "-f", "mp4", str(self.part)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.closed = False

    def append(self, array):
        if self.closed or array.shape != (768, 1344, 3):
            raise ValueError("full-resolution RGB frame required")
        self.process.stdin.write(array.tobytes())

    def finish(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            rc = self.process.wait(timeout=60)
            if rc:
                raise RuntimeError("archive encoder failed: " + str(rc))
            self.part.replace(self.output)
        finally:
            self.log.close()

    def abort(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        self.log.close()
