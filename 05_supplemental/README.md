# Extras: teaching PySR new words

Not part of the taught sessions. This is the answer sheet for the questions that arrive at the
end, the ones that start "can it also...", and it is meant to be worked through afterwards at
your own pace.

Everything runs on one small problem, a tracer diffusing into a bar, whose law is
`erfc(x / sqrt(t))`. Since erfc is not one of the operators PySR ships with, the problem keeps
handing us reasons to extend the tool.

What is in it:

1. the data, and the similarity collapse that says the answer has one variable
2. what a free search does without the right operator (a thirty-node nest)
3. writing your own operator in Julia, and the fitted constant turning out to be `2/sqrt(pi)`
4. borrowing `erfc` from SpecialFunctions.jl, which takes thirteen nodes down to five
5. why an operator has to be defined on the whole real line, and the typed-NaN guard
6. exports: sympy, latex, `latex_table`, and a differentiable PyTorch module
7. a `loss_function` that walks the expression tree and penalises structure
8. `complexity_of_constants` and the rest of the complexity-shaping knobs
9. two targets in one call
10. `warm_start`, to put a search down and pick it up again
11. saving and reloading, and the two ways a custom operator gets lost on the way back
12. `jl.seval` into the live Julia session, plus what PySR 2.0 changes

## Running it

```
python build_notebook.py            # regenerates 05_supplemental.ipynb; never hand-edit the json
jupyter nbconvert --to notebook --execute 05_supplemental.ipynb \
    --output 05_supplemental_EXECUTED.ipynb
```

`05_supplemental.ipynb` is the copy to work from, with outputs cleared.
`05_supplemental_EXECUTED.ipynb` is the reference copy, same sources, outputs kept.

About four minutes end to end on a laptop and rather more on Colab, most of it Julia compiling.
Section 4 adds `SpecialFunctions` to the Julia environment, which is another minute or so the
first time on a fresh machine. Checked against pysr 1.5.9 and sympy 1.14.0.
