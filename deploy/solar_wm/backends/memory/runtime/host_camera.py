"""Exact native navigation with append-only retrieval-camera CPU transfer."""
import ast
import copy
import inspect
import textwrap
import types
from deploy.solar_wm.backends.memory.runtime.retrieval_camera import RetrievalCamera

TAIL = '''view_cpu = self.inputs.camera_viewmats[0, 1:needed].cpu().numpy().astype(np.float64)
self.memory_forward.c2w = np.linalg.inv(view_cpu)
self.memory_forward.K = self.inputs.camera_K[0, 1:needed].cpu().numpy()
'''


def adapted_advance(native):
    source = textwrap.dedent(inspect.getsource(native))
    tree = ast.parse(source)
    fn = tree.body[0]
    tail = ast.parse(TAIL).body
    if [ast.dump(n) for n in fn.body[-3:]] != [ast.dump(n) for n in tail]:
        raise RuntimeError('retrieval tail differs')
    original = ast.dump(tree)
    saved = fn.body[-3:]
    fn.body[-3:] = ast.parse('''self.memory_forward.c2w, self.memory_forward.K = self.retrieval_camera.append(
    self.inputs, self.inputs.camera_viewmats, self.inputs.camera_K, start, needed)
''').body
    candidate = copy.deepcopy(tree)
    fn.body[-1:] = saved
    if ast.dump(tree) != original:
        raise RuntimeError('non-retrieval navigation changed')
    namespace = {}
    exec(compile(ast.fix_missing_locations(candidate), '<append-only-retrieval-camera>', 'exec'), native.__globals__, namespace)
    return namespace[fn.name]


def install_for_request(engine):
    """Bind append-only camera transfer to this request, not the engine class."""
    from deploy.solar_wm.backends.memory.runtime.memory_engine import MemoryEngine
    engine.retrieval_camera = RetrievalCamera()
    engine.advance_camera = types.MethodType(adapted_advance(MemoryEngine.advance_camera), engine)
