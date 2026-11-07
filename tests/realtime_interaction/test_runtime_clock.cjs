// node --test tests/realtime_interaction/test_runtime_clock.cjs
// CPU-only checks of the shipped player, with a controllable monotonic clock.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../deploy/solar_wm/web/static');
const source = fs.readFileSync(path.join(root, 'live_preview.js'), 'utf8');
const context = {module: {exports: {}}};
vm.runInNewContext(source.slice(0, source.indexOf('// Narrow demuxer')), context);
const {RuntimeClock} = context.module.exports;

function fixture() {
  let now = 100000;
  const values = new Map();
  const storage = {getItem: key => values.get(key), setItem: (key, value) => values.set(key, value)};
  const clock = new RuntimeClock(storage, () => now);
  const row = {deployment_id: 'release-test', scene_id: 'snow', stream_sequence: 1,
    active: true, paused: false, start_pending: false, chunk: 0};
  return {clock, row, storage, now: () => now, advance: ms => {now += ms;}};
}

test('sixty real seconds stay sixty seconds at any generation FPS', () => {
  for (const fps of [16, 23, 28, 50, 1000]) {
    const f = fixture(); f.clock.observe(f.row, {started: true});
    for (let i = 1; i <= 600; i++) {
      f.advance(100);
      f.clock.observe({...f.row, chunk: i, frames: i * fps, generated_fps: fps});
      for (let j = 0; j < 5; j++) f.clock.render({});
    }
    assert.equal(f.clock.snapshot().ms, 60000);
    const label = {}; f.clock.render(label); assert.equal(label.textContent, 'Runtime 01:00');
  }
});

test('duplicate status and render callbacks do not accumulate extra time', () => {
  const f = fixture(); f.clock.observe(f.row, {started: true});
  for (let i = 0; i < 100; i++) {
    f.advance(100);
    for (let j = 0; j < 30; j++) {f.clock.observe(f.row); f.clock.snapshot(); f.clock.render({});}
  }
  assert.equal(f.clock.snapshot().ms, 10000);
});

test('pause, resume and stop exclude pauses and freeze the completed timer', () => {
  const f = fixture(); f.clock.observe(f.row, {started: true});
  f.advance(2000); f.clock.observe({...f.row, paused: true});
  f.advance(40000); assert.equal(f.clock.snapshot().ms, 2000);
  f.clock.observe(f.row); f.advance(3000); f.clock.observe({...f.row, active: false});
  f.advance(60000); assert.equal(f.clock.snapshot().ms, 5000);
});

test('pending start preserves elapsed time when the next sequence is published', () => {
  const f = fixture();
  f.clock.observe({...f.row, stream_sequence: 0, active: false, start_pending: true}, {started: true});
  f.advance(1500); f.clock.observe(f.row); f.advance(500);
  assert.equal(f.clock.snapshot().ms, 2000);
});

test('late start acknowledgement does not reset a running timer', () => {
  const f = fixture(); f.clock.observe(f.row); f.advance(1000);
  f.clock.observe(f.row, {started: true}); assert.equal(f.clock.snapshot().ms, 1000);
});

test('new sessions, scenes and deployments start from zero', () => {
  for (const change of [{stream_sequence: 2}, {scene_id: 'fantasy_world'}, {deployment_id: 'another'}]) {
    const f = fixture(); f.clock.observe(f.row, {started: true});
    f.advance(3000); f.clock.observe({...f.row, active: false});
    f.clock.observe({...f.row, ...change}, {started: true});
    assert.equal(f.clock.snapshot().ms, 0);
  }
});

test('page reload restores a recently observed session', () => {
  const f = fixture(); f.clock.observe(f.row, {started: true});
  f.advance(2000); f.clock.observe(f.row); f.advance(500);
  const restored = new RuntimeClock(f.storage, f.now); restored.observe(f.row);
  assert.equal(restored.snapshot().ms, 2500);
});

test('late joining does not invent a start time', () => {
  const f = fixture(); f.clock.observe({...f.row, chunk: 100});
  const label = {}; f.clock.render(label); assert.equal(label.textContent, 'Runtime —');
});

test('unobserved lifecycle changes are not presented as exact timing', () => {
  const f = fixture(); f.clock.observe(f.row, {started: true}); f.advance(10000);
  assert.equal(f.clock.snapshot().approximate, true);
  f.clock.observe({...f.row, active: false}); assert.equal(f.clock.snapshot().known, false);
});

test('storage denial and a runtime error do not make the timer keep running', () => {
  let now = 0;
  const c = new RuntimeClock({getItem(){throw Error('denied');},setItem(){throw Error('denied');}}, () => now);
  const row = fixture().row; c.observe(row, {started: true}); now = 1000;
  c.observe({...row, error: 'failed'}); now = 3000; assert.equal(c.snapshot().ms, 1000);
});

test('the effective UI uses elapsed time instead of the playback timeline', () => {
  const navigation = source.slice(source.indexOf('function navigationUI'), source.indexOf('setInterval(navigationUI'));
  assert.match(navigation, /h3RuntimeClock.render/);
  assert.doesNotMatch(navigation, /currentTime|frames|sourceFps/);
  const post = source.slice(source.indexOf('async function h3Post'), source.indexOf('function h3OrderedControl'));
  assert.match(post, /h3RuntimeClock.observe\(data,\{started:true\}\)/);
});

test('the initial HTML describes real elapsed time, not the 16 FPS media clock', () => {
  const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
  const timer = html.match(/<div id="roamTimer"[^>]*>[^<]*<\/div>/)[0];
  assert.match(timer, /Elapsed real time/);
  assert.match(timer, />Runtime 00:00<\/div>/);
  assert.doesNotMatch(timer, /16 FPS|视频时间轴|漫游/);
});
