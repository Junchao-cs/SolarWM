"""Build operator bindings from this process's generated wrappers.

The recipe identifies supported kernel arithmetic, not a historical filename,
binary or machine. Unknown code is rejected before installing replacements.
All paths and binary identities are obtained from the current compilation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

from .schedule import transform


def signature(source):
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError('one generated kernel required')
    function = functions[0]
    signatures = [ast.literal_eval(value) for decorator in function.decorator_list
        if isinstance(decorator, ast.Call) for keyword in decorator.keywords
        if keyword.arg == 'triton_meta' and isinstance(keyword.value, ast.Dict)
        for key, value in zip(keyword.value.keys, keyword.value.values)
        if isinstance(key, ast.Constant) and key.value == 'signature']
    if len(signatures) != 1:
        raise ValueError('exact generated pointer/scalar signature required')
    function.name = 'kernel'
    function.decorator_list = []
    payload = ast.dump(function, include_attributes=False) + json.dumps(signatures[0], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe():
    return json.loads(Path(__file__).with_name('recipe.json').read_text())


def save(path, value):
    data = value if isinstance(value, str) else json.dumps(value, indent=2) + '\n'
    if path.exists():
        if path.read_text() != data:
            raise RuntimeError('operator build changed within its cache namespace: ' + str(path))
    else:
        with path.open('x') as output:
            output.write(data)
    return path


def scalar_arguments(wrapper, name, kind):
    """Resolve layout scalars from Runner.call's actual input argument.

    Symbol names and benchmark example values are not runtime contracts. The
    source-checked helper ABI identifies the prefix argument; all other scalars
    must derive from that same input. Track assignments in execution order and
    never look outside the runner's scope (notably benchmark_compiled_module).
    """
    prefix_slot = {'gather': 3, 'copy': 3, 'pack': 2}[kind]
    arities = {'gather': (6,), 'copy': (5,), 'pack': (5, 6)}[kind]
    tree = ast.parse(wrapper)
    runners = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runner']
    functions = [n for r in runners for n in r.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'call']
    if len(runners) != 1 or len(functions) != 1:
        raise ValueError('one actual Runner.call required')
    function = functions[0]
    if (len(function.args.args) != 2 or function.args.posonlyargs
            or function.args.vararg or function.args.kwarg or function.args.kwonlyargs):
        raise ValueError('expected boxed Runner.call(self, args)')
    boxed = function.args.args[1].arg
    values, roots, found = {}, set(), []
    unpacked = False

    def resolve(n):
        if isinstance(n, ast.Name):
            if values.get(n.id) is None:
                raise ValueError('unbound layout scalar: ' + n.id)
            return copy.deepcopy(values[n.id])
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return copy.deepcopy(n)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            return ast.BinOp(left=resolve(n.left), op=n.op, right=resolve(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return ast.UnaryOp(op=n.op, operand=resolve(n.operand))
        raise ValueError('unexpected layout scalar dataflow')

    def matches(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'run' and isinstance(n.func.value, ast.Name)
                and n.func.value.id == name)

    def inspect_call(n):
        if len(n.args) not in arities or [k.arg for k in n.keywords] != ['stream']:
            raise ValueError('layout launch ABI changed')
        prefix = resolve(n.args[prefix_slot])
        if not isinstance(prefix, ast.Name) or prefix.id not in roots:
            raise ValueError('layout prefix must come from a runtime input')

        class Normalize(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id != prefix.id:
                    raise ValueError('layout scalar depends on another runtime input')
                return ast.Name(id='L', ctx=ast.Load())

        expressions = [ast.unparse(Normalize().visit(resolve(v))) for v in n.args[2:]]
        found.append((prefix.id, expressions))

    def walk(body):
        nonlocal unpacked
        for statement in body:
            if isinstance(statement, ast.With):
                # Generated device guards don't define scalar locals.
                if any(item.optional_vars is not None for item in statement.items):
                    raise ValueError('unsupported runner context binding')
                walk(statement.body)
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                values[statement.name] = None
                continue
            if isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.AsyncWith)):
                if any(matches(n) for n in ast.walk(statement)):
                    raise ValueError('conditional layout launch is unsupported')
                for n in ast.walk(statement):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                        values[n.id] = None
                continue
            for n in ast.walk(statement):
                if matches(n):
                    inspect_call(n)
            if isinstance(statement, ast.Assign):
                if (len(statement.targets) == 1
                        and isinstance(statement.targets[0], (ast.Tuple, ast.List))
                        and isinstance(statement.value, ast.Name) and statement.value.id == boxed):
                    targets = statement.targets[0].elts
                    if (unpacked or statement is not function.body[0]
                            or not all(isinstance(n, ast.Name) for n in targets)):
                        raise ValueError('ambiguous runtime input unpacking')
                    names = [n.id for n in targets]
                    if len(set(names)) != len(names) or boxed in names:
                        raise ValueError('aliased runtime input unpacking')
                    for index, target in enumerate(names):
                        root = '_input_' + str(index)
                        roots.add(root)
                        values[target] = ast.Name(id=root, ctx=ast.Load())
                    unpacked = True
                else:
                    try:
                        value = resolve(statement.value)
                    except ValueError:
                        value = None  # Tensor expressions aren't scalar definitions.
                    for target in statement.targets:
                        for n in ast.walk(target):
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                                values[n.id] = value if isinstance(target, ast.Name) else None
            elif isinstance(statement, (ast.AugAssign, ast.AnnAssign, ast.Delete)):
                for n in ast.walk(statement):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                        values[n.id] = None

    walk(function.body)
    if not found or any(x != found[0] for x in found):
        raise ValueError('missing or inconsistent layout calls')
    return found[0][1]


def triplets(source, records):
    functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == 'call']
    if len(functions) != 1:
        raise ValueError('one actual Runner.call required')
    calls = sorted((n for n in ast.walk(functions[0]) if isinstance(n, ast.Call)),
                   key=lambda n: (n.lineno, n.col_offset))
    kernels = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'run'
               and isinstance(n.func.value, ast.Name)]
    dump = lambda n: ast.dump(n, include_attributes=False)
    plans = {}
    for a, b, c in zip(kernels, kernels[1:], kernels[2:]):
        names = [n.func.value.id for n in (a, b, c)]
        if any(n not in records for n in names):
            continue
        if [records[n]['kind'] for n in names] != ['residual', 'reduce', 'pointwise']:
            continue
        if dump(b.args[1]) != dump(c.args[1]):
            raise ValueError('RMS reduction consumer changed')
        between = [n for n in calls if a.lineno < n.lineno < c.lineno and n is not b]
        if any(ast.unparse(n.func) not in ('empty_strided_cuda', 'reinterpret_tensor', 'get_raw_stream') for n in between):
            raise ValueError('side effect inside fusion triplet')
        if any([k.arg for k in n.keywords] != ['stream'] for n in (a, b, c)):
            raise ValueError('fusion stream keyword changed')
        if len({dump(n.keywords[0].value) for n in (a, b, c)}) != 1:
            raise ValueError('fusion streams differ')
        plans[a.lineno] = dict(functions=names, lines=[n.lineno for n in (a, b, c)],
            residual=records[names[0]]['contract'], reduction=records[names[1]]['contract'],
            pointwise=records[names[2]]['contract'])
    return plans


def build(source_path, helpers, *, phase, rank, directory):
    """CPU-only source checks; helpers map actual global names to source files."""
    if phase not in ('denoise', 'refresh') or rank not in range(8):
        raise ValueError('steady SP8 phase required')
    spec = recipe()
    source_path = Path(source_path)
    source = source_path.read_text()
    records, layouts, norms = {}, {}, []
    evidence = {str(source_path): sha(source_path)}
    for name, path in helpers.items():
        path = Path(path)
        text = path.read_text()
        key = signature(text)
        expected = spec['kernels'].get(key)
        if expected:
            record = dict(copy.deepcopy(expected), name=name, source=str(path), source_sha256=sha(path))
            evidence[str(path)] = record['source_sha256']
            if record.pop('role') == 'fma':
                records[name] = record
            else:
                record['scalar_expressions'] = scalar_arguments(source, name, record['kind'])
                layouts[path.name] = record
        for rule in spec['norms']:
            if rule['name'] == 'rank_fma_alignment':
                # This kernel belongs to the pre-steady history regimes.
                continue
            if rule['name'] == 'denoise_normalization' and phase != 'denoise':
                continue
            if rule['name'] == 'refresh_normalization' and phase != 'refresh':
                continue
            if rule['rank'] == rank and rule['signature'] == key and rule['function'] == name:
                if rule['first_refresh_call_only'] and phase != 'refresh':
                    continue
                norms.append(dict(rule, helper=name, source=str(path), source_sha256=sha(path)))
    plans = triplets(source, records)
    expected_count = spec['phases'][phase][rank]
    if len(plans) != expected_count:
        raise RuntimeError(f'{phase} rank {rank}: supported fusion sites {len(plans)} != {expected_count}')
    if any(p['pointwise']['bound'] != (12 if phase == 'denoise' else 9) for p in plans.values()):
        raise RuntimeError('fusion modulation extent belongs to another phase')
    if len(layouts) != 4 or sorted(r['kind'] for r in layouts.values()) != ['copy', 'gather', 'pack', 'pack']:
        raise RuntimeError('all four supported SP layout helpers are required')
    expected_norms = 1 if rank == 5 else int(phase == 'refresh' and rank in (0, 1))
    if len(norms) != expected_norms:
        raise RuntimeError(f'{phase} rank {rank}: missing or additional normalization configuration')
    directory = Path(directory) / phase
    directory.mkdir(parents=True, exist_ok=True)
    row = dict(rank=rank, wrapper=str(source_path), wrapper_sha256=sha(source_path), records=records, plans=plans)
    kernel_file = Path(__file__).with_name('rms_fma_kernels.py' if phase == 'denoise' else 'rms_fma_refresh_kernels.py')
    fma = save(directory / 'fusion.json', dict(phase=phase, index_bound=12 if phase == 'denoise' else 9,
        ranks=[row if i == rank else None for i in range(8)], evidence_sha256=evidence,
        kernels=str(kernel_file), kernels_sha256=sha(kernel_file)))
    layout = save(directory / 'layout.json', dict(ranks=[dict(records=layouts) if i == rank else None for i in range(8)],
        evidence_sha256=evidence))
    result = dict(fma=fma, layout=layout, norms=norms, plans=plans, source=str(source_path))
    if phase == 'denoise':
        fn, receipt = transform(source)
        candidate = save(directory / 'scheduled.py', ast.unparse(fn) + '\n')
        receipt.update(rank=rank, original=str(source_path), candidate=str(candidate))
        result['qkv'] = save(directory / 'qkv.json', dict(ranks=[receipt if i == rank else None for i in range(8)]))
    return result
