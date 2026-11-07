"""CPU-only regressions for generated layout scalar provenance."""
import unittest

from deploy.solar_wm.backends.standard.acceleration.operators.build import scalar_arguments
from deploy.solar_wm.backends.standard.acceleration.operators.layout import dimensions


def wrapper(symbol="s99", kind="gather", rank=0, extra="", tail=""):
    prefix = symbol
    # The production expression grammar uses integer arithmetic, not compares.
    extent = f"7168 * ((5573 - {rank} + {prefix}) // 8)"
    args = {
        "gather": f"x, y, total, {prefix}, width, count",
        "pack": f"x, y, {prefix}, 128 * {prefix}, count",
        "copy": f"x, y, 128 * ((5573 - {rank} + {prefix}) // 8), {prefix}, {extent}",
    }[kind]
    return f"""
class Runner:
    def call(self, args):
        x, y, arg83_1, another = args
        {symbol} = arg83_1
        total = 5566 + {symbol}
        width = 712448 + 128 * {symbol}
        count = 4987136 + 896 * {symbol}
        {extra}
        with device_guard(0):
            helper.run({args}, stream=stream0)
            {tail}

def benchmark_compiled_module():
    arg83_1 = 2123
    s100 = 2123
    helper.run(x, y, 1, 2, 3, 4, stream=stream0)
"""


class LayoutBindingTests(unittest.TestCase):
    def test_symbol_names_are_not_an_abi(self):
        for name in ("s99", "s100", "s0", "arbitrary_prefix"):
            with self.subTest(name=name):
                self.assertEqual(scalar_arguments(wrapper(name), "helper", "gather"),
                    ["5566 + L", "L", "712448 + 128 * L", "4987136 + 896 * L"])

    def test_all_lengths_ranks_and_layout_kinds(self):
        # Includes both endpoints, the reported 1140-token failure, and every
        # uneven/even SP partition residue. No tensors or CUDA are involved.
        for kind in ("gather", "pack", "copy"):
            for rank in range(8):
                expressions = scalar_arguments(wrapper(kind=kind, rank=rank), "helper", kind)
                record = dict(kind=kind, scalar_expressions=expressions)
                for length in range(1000, 1301):
                    prefix = length + 1008
                    total = 5566 + prefix
                    count = 7168 * ((total + 7 - rank) // 8) if kind == "copy" else 896 * total
                    if kind == "gather":
                        scalars = (total, prefix, 128 * total, count)
                    elif kind == "copy":
                        scalars = (128 * ((total + 7 - rank) // 8), prefix, count)
                    else:
                        scalars = (prefix, 128 * prefix, count)
                    self.assertEqual(dimensions(record, rank, (None, None, *scalars)),
                                     (length, total, count))

    def test_alias_chain(self):
        text = wrapper().replace("s99 = arg83_1", "first = arg83_1\n        s99 = first")
        self.assertEqual(scalar_arguments(text, "helper", "gather")[1], "L")

    def test_definitions_after_call_do_not_change_earlier_values(self):
        text = wrapper(tail="s99 = 9999")
        self.assertEqual(scalar_arguments(text, "helper", "gather")[1], "L")

    def test_nested_function_and_benchmark_are_not_runtime_scope(self):
        text = wrapper().replace("        with device_guard(0):", """        def nested():
            arg83_1 = 4444
            helper.run(x, y, 1, 2, 3, 4, stream=stream0)
        with device_guard(0):""")
        self.assertEqual(scalar_arguments(text, "helper", "gather")[1], "L")

    def test_secondary_pack_abi(self):
        text = wrapper(kind="pack").replace("x, y, s99, 128 * s99, count",
            "x, y, s99, 128 * ((5573 + s99) // 8), 128 * s99, count")
        self.assertEqual(len(scalar_arguments(text, "helper", "pack")), 4)

    def test_invalid_bindings_rejected(self):
        variants = {
            "constant prefix": wrapper().replace("s99 = arg83_1", "s99 = 2123"),
            "global prefix": wrapper().replace("s99 = arg83_1", "s99 = missing"),
            "another input": wrapper().replace("width = 712448 + 128 * s99", "width = another"),
            "prefix expression": wrapper().replace("s99 = arg83_1", "s99 = arg83_1 + 1"),
            "overwrite before use": wrapper(extra="s99 = 2123"),
            "unknown keyword": wrapper().replace("stream=stream0", "other=stream0"),
            "wrong arity": wrapper().replace("x, y, total, s99, width, count", "x, y, s99"),
            "wrong function": wrapper().replace("def call(self, args):", "def unrelated(self, args):"),
            "wrong unpack": wrapper().replace("x, y, arg83_1, another = args", "x, y, arg83_1, another = unknown"),
            "boxed reassignment": wrapper().replace("        x, y, arg83_1, another = args", "        args = other_args\n        x, y, arg83_1, another = args"),
            "duplicate input": wrapper().replace("x, y, arg83_1, another = args", "x, y, arg83_1, arg83_1 = args"),
            "conditional launch": wrapper().replace("with device_guard(0):", "if flag:"),
            "conditional scalar": wrapper(extra="if flag: s99 = another"),
            "mutated scalar": wrapper(extra="s99 += 1"),
            "deleted scalar": wrapper(extra="del s99"),
            "context alias": wrapper().replace("with device_guard(0):", "with device_guard(0) as s99:"),
            "different second call": wrapper(tail="helper.run(x, y, total, s99, width + 1, count, stream=stream0)"),
            "different input second call": wrapper(tail="helper.run(x, y, 5566 + another, another, 712448 + 128 * another, 4987136 + 896 * another, stream=stream0)"),
        }
        for label, text in variants.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                scalar_arguments(text, "helper", "gather")

    def test_runtime_checks_not_relaxed(self):
        record = dict(kind="gather", scalar_expressions=scalar_arguments(wrapper(), "helper", "gather"))
        prefix = 2148
        args = [None, None, 5566 + prefix, prefix, 712448 + 128 * prefix, 4987136 + 896 * prefix]
        for index in range(2, 6):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                broken = args.copy()
                broken[index] += 1
                dimensions(record, 0, broken)


if __name__ == "__main__":
    unittest.main()
