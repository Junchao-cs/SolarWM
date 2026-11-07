"""Move only independent K/V GEMMs before Q/K waits in observed H3 wrappers.

No wait or collective is removed/reordered. Existing mm/Triton calls remain
byte-for-AST identical. One reuse of an in-flight send allocation per moved
projection is replaced by fresh storage, and its send reference is retained
through the original wait. This is a source-bound candidate, not a general
Python scheduler. Runtime correctness and memory cost require a short SP8 test.
"""
import ast
import copy
import hashlib
from collections import Counter
from pathlib import Path

COMM = 'torch.ops._c10d_functional.all_to_all_single.default'
WAIT = 'torch.ops._c10d_functional.wait_tensor.default'
MM = 'extern_kernels.mm'


def dump(n):
    return ast.dump(n, include_attributes=False)


def calls(n, name):
    return [c for c in ast.walk(n) if isinstance(c, ast.Call) and ast.unparse(c.func) == name]


def names(n, context=ast.Load):
    return {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, context)}


def tensor_name(name):
    return name.startswith('buf') or name.startswith('arg')


def assigned(n):
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
        return n.targets[0].id
    return None


def alias_source(value):
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call) and ast.unparse(value.func) == 'reinterpret_tensor':
        if len(value.args) != 4 or not isinstance(value.args[0], ast.Name):
            raise ValueError('unexpected reinterpret ABI')
        return value.args[0].id
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        return value.value.id
    return None


def inspect_storage(body):
    """Conservative whole-storage aliases, never disjoint-slice assumptions."""
    roots = {}
    info = []
    lifetimes = {}
    allocations = {}
    for i, node in enumerate(body):
        used = names(node)
        before_roots = {n: roots.get(n, 'external:' + n) for n in used if tensor_name(n)}
        target = assigned(node)
        if target and tensor_name(target):
            if target in lifetimes:
                raise ValueError('tensor names must be single assignment: ' + target)
            source = alias_source(node.value)
            root = roots.get(source, 'external:' + source) if source else 'allocated:' + target
            roots[target] = root
            lifetimes[target] = [i, len(body), root]
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'empty_strided_cuda':
                allocations[root] = node.value
            elif isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == COMM:
                # all_to_all preserves dtype even when its newly allocated output
                # is reused by a later layer. Shape is taken from the actual
                # zero-offset reinterpret below, never this ancestor allocation.
                send = node.value.args[0]
                if isinstance(send, ast.Name):
                    origin = before_roots.get(send.id)
                    if origin in allocations:
                        allocations[root] = allocations[origin]
        deleted = names(node, ast.Del)
        for name in deleted:
            if name in lifetimes:
                lifetimes[name][1] = i
            roots.pop(name, None)
        mm = calls(node, MM)
        writes = set()
        if mm:
            if len(mm) != 1:
                raise ValueError('multiple GEMMs in statement')
            outs = [k.value for k in mm[0].keywords if k.arg == 'out']
            if len(outs) != 1 or not isinstance(outs[0], ast.Name):
                raise ValueError('GEMM output ABI')
            writes.add(roots.get(outs[0].id, 'external:' + outs[0].id))
        else:
            # Calls other than pure allocation/metadata can mutate any tensor
            # argument. This deliberately rejects unproven alias independence.
            for call in [c for c in ast.walk(node) if isinstance(c, ast.Call)]:
                callee = ast.unparse(call.func)
                if callee not in ('empty_strided_cuda', 'reinterpret_tensor', 'assert_size_stride',
                                  'assert_alignment', 'get_raw_stream'):
                    writes.update(before_roots.values())
        info.append(dict(used=used, defined=names(node, ast.Store), deleted=deleted,
                         roots=before_roots, writes=writes))
    return info, lifetimes, allocations


def _projection(body, info, lifetimes, allocations, launch_index, wait_index, next_launch):
    launch = calls(body[launch_index], COMM)[0]
    send = launch.args[0]
    if not isinstance(send, ast.Name):
        raise ValueError('send must be an explicit buffer')
    send = send.id
    send_root = info[launch_index]['roots'][send]
    mm_indices = [i for i in range(wait_index + 1, next_launch) if calls(body[i], MM)]
    if len(mm_indices) != 3:
        raise ValueError('expected exactly three K/V projection GEMMs after wait')
    first_mm = calls(body[mm_indices[0]], MM)[0]
    first_out = next(k.value.id for k in first_mm.keywords if k.arg == 'out')
    starts = [i for i in range(wait_index + 1, mm_indices[0]) if assigned(body[i]) == first_out]
    if len(starts) != 1:
        raise ValueError('projection allocation not uniquely found')
    start, end = starts[0], mm_indices[-1] + 1
    block = body[start:end]
    for node in block:
        if isinstance(node, ast.Delete) and all(isinstance(x, ast.Name) for x in node.targets):
            continue
        if calls(node, MM) and isinstance(node, ast.Expr):
            if len(calls(node, MM)) == 1:
                continue
        if assigned(node):
            value = node.value
            if isinstance(value, ast.Name):
                continue
            if isinstance(value, ast.Call) and ast.unparse(value.func) in ('reinterpret_tensor', 'empty_strided_cuda'):
                continue
        raise ValueError('non-projection operation inside moved block')
    aliases = [(i, assigned(body[i])) for i in range(start, end) if assigned(body[i]) and
               lifetimes.get(assigned(body[i]), [None, None, None])[2] == send_root]
    if len(aliases) != 1:
        raise ValueError('expected one in-flight send-storage reuse')
    reuse_index, output = aliases[0]
    original = body[reuse_index]
    value = original.value
    if not (isinstance(value, ast.Call) and ast.unparse(value.func) == 'reinterpret_tensor' and
            isinstance(value.args[0], ast.Name) and value.args[0].id == send and
            isinstance(value.args[3], ast.Constant) and value.args[3].value == 0):
        raise ValueError('send reuse must be zero-offset reinterpret')
    allocation = allocations.get(send_root)
    if allocation is None or len(allocation.args) != 3 or ast.unparse(allocation.args[2]) != 'torch.bfloat16':
        raise ValueError('send allocation must be observed BF16 storage')
    live_after = {n for n, (a, b, r) in lifetimes.items() if a < end <= b and r == send_root}
    if live_after != {output}:
        raise ValueError('unhandled surviving alias of replaced send storage')
    releases = [i for i in range(start, end) if isinstance(body[i], ast.Delete) and
                names(body[i], ast.Del) == {send}]
    if len(releases) != 1:
        raise ValueError('send lifetime release not unique')
    fresh = copy.deepcopy(original)
    fresh.value = ast.Call(func=ast.Name(id='empty_strided_cuda', ctx=ast.Load()),
                          args=[copy.deepcopy(value.args[1]), copy.deepcopy(value.args[2]),
                                copy.deepcopy(allocation.args[2])], keywords=[])
    moved = [fresh if i == reuse_index else copy.deepcopy(body[i]) for i in range(start, end)
             if i != releases[0]]
    crossed = info[wait_index:start]
    moved_info = info[start:end]
    moved_writes = set().union(*(r['writes'] for r in moved_info)) - {send_root}
    moved_reads = set().union(*(set(r['roots'].values()) for r in moved_info)) - {send_root}
    crossed_roots = set().union(*(set(r['roots'].values()) for r in crossed))
    crossed_writes = set().union(*(r['writes'] for r in crossed))
    if moved_writes & crossed_roots or moved_reads & crossed_writes:
        raise ValueError('storage dependency across proposed projection movement')
    deleted = set().union(*(r['deleted'] for r in moved_info)) - {send}
    crossed_used = set().union(*(r['used'] for r in crossed))
    crossed_defined = set().union(*(r['defined'] for r in crossed))
    crossed_deleted = set().union(*(r['deleted'] for r in crossed))
    moved_used = set().union(*(r['used'] for r in moved_info))
    if deleted & crossed_used or moved_used & (crossed_defined | crossed_deleted):
        raise ValueError('Python binding/lifetime dependency across movement')
    return dict(start=start, end=end, before=wait_index, moved=moved,
                release=copy.deepcopy(body[releases[0]]),
                receipt=dict(send=send, new_storage=output, first_line=body[start].lineno,
                             last_line=body[end-1].end_lineno, before_wait_line=body[wait_index].lineno,
                             moved_mm=3, allocation_shape=ast.unparse(value.args[1]),
                             allocation_stride=ast.unparse(value.args[2]), dtype='bfloat16'))


def _pure_scalar(node):
    return all(isinstance(n, (ast.Constant, ast.Name, ast.Load, ast.BinOp,
        ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
        ast.USub, ast.UAdd, ast.Tuple)) for n in ast.walk(node))


def _check_block(nodes):
    for node in nodes:
        if isinstance(node, ast.Delete) and all(isinstance(t, ast.Name) for t in node.targets):
            continue
        if assigned(node):
            value = node.value
            if _pure_scalar(value):
                continue
            if isinstance(value, ast.Call) and ast.unparse(value.func) in (
                    'reinterpret_tensor', 'empty_strided_cuda', 'get_raw_stream', COMM):
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name in (MM, 'assert_size_stride', 'assert_alignment') or name.endswith('.run'):
                continue
        raise ValueError('unrecognized full projection/preparation operation')


def _stable_stream(body):
    # Generated wrappers name the current device stream stream0..stream7.
    # Require an exact matching device literal and identical definitions,
    # instead of making rank zero's spelling part of the scheduling contract.
    bindings = {assigned(n) for n in body if assigned(n) and
                assigned(n).startswith('stream')}
    stable = set()
    for name in bindings:
        suffix = name.removeprefix('stream')
        if not suffix.isdigit() or not 0 <= int(suffix) < 8:
            raise ValueError('unexpected stream binding')
        definitions = [n for n in body if assigned(n) == name]
        values = {dump(n.value) for n in definitions}
        if len(values) != 1 or not all(isinstance(n.value, ast.Call) and
                ast.unparse(n.value.func) == 'get_raw_stream' and
                len(n.value.args) == 1 and isinstance(n.value.args[0], ast.Constant)
                and type(n.value.args[0].value) is int
                and n.value.args[0].value == int(suffix)
                and not n.value.keywords for n in definitions):
            raise ValueError('stream identity is not fixed throughout the device scope')
        if any(name in names(n, ast.Del) for n in body):
            raise ValueError('stream binding deleted')
        stable.add(name)
    if len(stable) > 1:
        raise ValueError('multiple device streams in single scheduling scope')
    return stable


def _independent(info, plan, crossed_indices, stable):
    moved = info[plan['start']:plan['end']]
    crossed = [info[i] for i in crossed_indices]
    union = lambda rs, key: set().union(*(r[key] for r in rs))
    roots = lambda rs: set().union(*(set(r['roots'].values()) for r in rs))
    # The only split alias is source-checked and allocated afresh. All other
    # roots are treated as whole storage; no non-overlapping slice assumptions.
    replaced = {plan['send_root']}
    writes = union(moved, 'writes') - replaced
    reads = roots(moved) - replaced
    if writes & roots(crossed) or reads & union(crossed, 'writes'):
        raise ValueError('full pipeline storage dependency across movement')
    deleted = union(moved, 'deleted') - {plan['receipt']['send']}
    used = union(moved, 'used') - stable
    defined = union(moved, 'defined') - stable
    cross_used = union(crossed, 'used') - stable
    cross_bound = (union(crossed, 'defined') | union(crossed, 'deleted')) - stable
    if deleted & cross_used or used & cross_bound or defined & cross_used:
        raise ValueError('full pipeline Python binding/lifetime dependency')

def _scalar_hoists(body, info, pair, before, stable):
    moved_indices = {i for p in pair for i in range(p['start'], p['end'])}
    candidates = set()
    for p in pair:
        crossed = [i for i in range(before,p['start']) if i not in moved_indices]
        used = set().union(*(x['used'] for x in info[p['start']:p['end']])) - stable
        bound = set().union(*(info[i]['defined'] | info[i]['deleted'] for i in crossed))
        for name in used & bound:
            found = [i for i in crossed if assigned(body[i]) == name]
            if len(found) != 1:
                raise ValueError('metadata dependency is not one scalar definition')
            i = found[0]
            node = body[i]
            if tensor_name(name) or not _pure_scalar(node.value) or any(
                    tensor_name(n) for n in names(node.value)):
                raise ValueError('cannot hoist tensor-dependent metadata')
            prior = info[before:i]
            dependencies = names(node.value)
            if dependencies & set().union(*(x['defined'] | x['deleted'] for x in prior)):
                raise ValueError('scalar metadata operand changes across hoist')
            if name in set().union(*(x['used'] | x['deleted'] | x['defined'] for x in prior)):
                raise ValueError('scalar metadata binding used before its original definition')
            candidates.add(i)
    return sorted(candidates)


def transform(source, *, expected_layers=50):
    tree = ast.parse(source)
    runners = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runner']
    if len(runners) != 1:
        raise ValueError('one observed Runner required')
    functions = [n for n in runners[0].body if isinstance(n, ast.FunctionDef) and n.name == 'call']
    if len(functions) != 1:
        raise ValueError('one Runner.call required')
    function = functions[0]
    scopes = [n for n in ast.walk(function) if isinstance(n, ast.With) and
        sum(bool(calls(s, COMM)) for s in n.body) == expected_layers * 4]
    if len(scopes) != 1:
        raise ValueError('one flat full-model collective scope required')
    scope = scopes[0]
    body = scope.body
    if any(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With)) for n in body):
        raise ValueError('nested control flow not supported')
    launches = [i for i,n in enumerate(body) if calls(n, COMM)]
    waits = [i for i,n in enumerate(body) if calls(n, WAIT)]
    if len(waits) not in (len(launches), len(launches)+2):
        raise ValueError('collective/wait count changed')
    info, lifetimes, allocations = inspect_storage(body)
    stable = _stable_stream(body)
    plans = []
    insert = {}
    releases = {}
    hoists = {}
    for layer in range(expected_layers):
        group = launches[layer*4:layer*4+4]
        matched = []
        for a,b in zip(group, group[1:]):
            found = [i for i in waits if a < i < b]
            if len(found) != 1:
                raise ValueError('expected one wait between original Q/K/V/output launches')
            wait = calls(body[found[0]], WAIT)[0]
            if len(wait.args) != 1 or ast.unparse(wait.args[0]) != assigned(body[a]):
                raise ValueError('wait does not match preceding collective')
            matched.append(found[0])
        pair=[]
        for j in (0,1):
            # Reuse the established strict send-allocation/lifetime proof, but
            # extend the move through the next collective and metadata checks.
            p = _projection(body, info, lifetimes, allocations, group[j], matched[j], group[j+1])
            partial_end = p['end']
            p['end'] = matched[j+1]
            _check_block(body[p['start']:p['end']])
            p['moved'].extend(copy.deepcopy(body[partial_end:p['end']]))
            p['send_root'] = info[group[j]]['roots'][p['receipt']['send']]
            p['receipt'].update(layer=layer, projection='K' if j == 0 else 'V',
                last_line=body[p['end']-1].end_lineno,
                before_wait_line=body[matched[0]].lineno, moved_collectives=1,
                preserved_wait_line=body[matched[j]].lineno)
            pair.append(p)
        # K remains before V. V must cross only instructions left behind,
        # not the K preparation that moves ahead of it as part of this pair.
        lifted = _scalar_hoists(body,info,pair,matched[0],stable)
        hoists[matched[0]] = lifted
        for p in pair:
            crossed = [i for i in range(matched[0],p['start'])
                if i not in lifted and not any(other['start'] <= i < other['end'] for other in pair)]
            try:
                _independent(info,p,crossed,stable)
            except ValueError as error:
                raise ValueError(f'layer {layer} {p["receipt"]["projection"]}: {error}') from error
            releases[p['before']] = p['release']
        insert[matched[0]] = pair
        plans.extend(pair)
    skipped = {i for p in plans for i in range(p['start'],p['end'])}
    if len(skipped) != sum(p['end']-p['start'] for p in plans) or skipped & (set(insert)|set(releases)):
        raise ValueError('overlapping pipeline edits')
    lifted_indices = {i for indices in hoists.values() for i in indices}
    if lifted_indices & skipped or len(lifted_indices) != sum(map(len,hoists.values())):
        raise ValueError('overlapping scalar hoists')
    skipped |= lifted_indices
    rewritten=[]
    for i,node in enumerate(body):
        if i in skipped:
            continue
        rewritten.extend(copy.deepcopy(body[j]) for j in hoists.get(i,[]))
        for p in insert.get(i,[]):
            rewritten.extend(p['moved'])
        rewritten.append(node)
        if i in releases:
            rewritten.append(releases[i])
    original=copy.deepcopy(function)
    scope.body=rewritten
    for callee in (COMM,WAIT):
        if [dump(c) for c in calls(original,callee)] != [dump(c) for c in calls(function,callee)]:
            raise ValueError('collective/wait sequence changed')
    def math(fn):
        return Counter(dump(c) for c in ast.walk(fn) if isinstance(c,ast.Call) and
            (ast.unparse(c.func).endswith('.run') or ast.unparse(c.func).startswith('extern_kernels.')
             or ast.unparse(c.func).startswith('torch.ops.')))
    if math(original) != math(function):
        raise ValueError('mathematical call altered')
    function=ast.fix_missing_locations(function)
    rendered=ast.unparse(function)+'\n'
    return function,dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_function_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        layers=expected_layers,moved_gemms=6*expected_layers,
        moved_collectives=2*expected_layers,fresh_send_reuse_allocations=2*expected_layers,
        collectives_unchanged=len(launches),waits_unchanged=len(waits),
        scalar_metadata_hoists=[dict(name=assigned(body[i]),expression=ast.unparse(body[i].value),line=body[i].lineno) for i in sorted(lifted_indices)],
        kernel_math_unchanged=True,windows=[p['receipt'] for p in plans])
