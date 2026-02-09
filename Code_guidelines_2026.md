# Coding Guidelines COP 2026

**Matthieu Schaller**

## Introduction

A lot of coding standards come from the fact that, while people can generally calculate things well, our working memory is extremely small—about 7 ± 2 things. On a bad day this means you can only keep track of five things, meaning that large computations are generally out of the question. To counter this, one can put parts of the computation behind a (sometimes abstract) concept. Think of pressure, you don't think of that in terms of the change of momentum of a number of particles hitting a certain area (which would take 4 slots of your valuable working memory), but rather as a concept by itself (which only takes one slot of working memory). In programming, this abstracting ranges from putting things in functions to well-chosen names for both variables and these functions.

Programming is not an exact science; some would say it's an art or like gardening. A program can be formatted in many different ways, in different programming paradigms (think imperative, object-oriented, functional etc), and do the same thing. You decide what the best possible way to write and structure your program is.

This document consists of two sections: the first section contains all **mandatory guidelines**, meaning points will be subtracted from your grade if not adhered to; and the second section will contain **optional guidelines**, which are either difficult to grade, up for interpretation or a matter of personal preference.

---

## Mandatory Code Guidelines

### Submission Requirements

- **Jupyter notebooks are awesome for testing and development. They are terrible to distribute code.** When submitting your project, do copy your code into stand-alone scripts with `.py` extensions instead. Jupyter notebooks will **NOT** be considered a valid submission for grading.

- **Provide a README file.** In it explain how to run your code; how to make your plots. Explain what the parameters are (if there are any) or what are the things to change if one want to try a different simulation (for instance more or less particles; higher/lower temperature, etc.)

### Naming

- **Variables should be longer than 1 character**, except counters in for loops. For example, don't use `v` for velocity, but use `velocity` or `vel` instead. Can you really infer what a variable contains just from its name? Without context?

- **Name functions appropriately.** The name should reflect what the function does, and ideally give some hints as to what it returns—if anything. For example, what would `distance(x, y)` return? What about `integrate_eqn_of_motion(timestep)`?

- **Function names should not include superfluous verbs**, such as 'calculate', 'compute', 'do'. A computer already computes...

- **Variable names should not include the type of the variable.** Make array and list names plural if appropriate.

- **Names should be consistent.** Stick to one particular scheme: `snake_case`, `camelCase`, `PascalCase`, or something similar.

### Functions

- **Functions should ideally do one thing.** In some cases this is easy, to calculate the force, for example. In other cases this might be more difficult, and the thing done by the function is more semantic, for example integrating the equation of motion to advance the simulation by a timestep. This latter function might contain other functions (with well-chosen names).

- **Ideally, don't repeat yourself.** If you see the same code more than once, think about whether it should actually be in a function.

### Documentation

- **Your Python code must contain docstrings.** These are strings just below the `def function_name` line that explain what the function does, and—if you don't use type hints—say what type the function returns, and the types of the arguments.

- **Ideally, your Python code also includes a docstring at the top of the file**, explaining what this file does, who wrote it, etc.

- **Comments.** The less comments you need the better. If you need them, don't explain what the code does (that should be clear if you abide by the rules above) but rather explain what the idea behind the code is. This can also be done in the docstring of a function.

### Structure

This section mixes structure for the entire program with the structure in functions.

- **Don't use global variables if possible.** It is always possible not to.

- **You should be able to call an instance (or sometimes many) of your simulation in another script** with particular initial conditions and parameters.

- **Decouple as much as possible.** For example, make sure your simulation can run without plotting.

### Optimisation

- **Do not try and optimise using bitwise operators.**

---

## Optional Code Guidelines

*(aka. recommendations & good practice for every day life as a programmer)*

### Organisation

Version control softwares, such as git, and platforms hosting git servers (like github or bitbucket) are extremely useful to share code within your group, to keep track of your changes, and roll back to earlier version if/when bugs are identified. Making use of such tools is recommended.

Note that you must submit a final version of your code. You cannot just submit a link to a github page.

### Naming

- **Generally avoid using capitals for variables and functions**, but use capitals for class names.

### Code Formatting

- **For python programs, running the tool `black` on your source code** is a good way to get a consistent format.

### Functions

- **Functions should take as few arguments as possible.**

- **Preferably, it should be clear from context what arguments a function takes.** If not, you can force users to use keyword arguments, by putting an asterisk (`*`) before all arguments you want to enforce with keywords. E.g.
  ```python
  def function_w_lotsa_args(arg1, arg2, *, kwarg1, kwarg2):
      pass
  ```
  is called as:
  ```python
  function_w_lotsa_args(1., 5., kwarg1=6., kwarg2=3.)
  ```

### Data

- **Think about what data structure would be appropriate for your data, and which would be fast.**

- **Very often the standard types** (int, float, double, arrays, etc) will do for 99% of your data. Non-standard data types, unless very carefully chosen, will take up your precious (human) working memory.

- **Do you want to store the current state of the system, or save it right away?** Opening, closing, and writing files on the computer is very slow. Don't do it during simulations, definitely don't do it in loops.

- **Object-oriented programming and classes can be helpful** in abstracting your data. However, don't go overboard with the classes. We suggest only to use classes if it makes a lot of sense and some pieces of data should always be close together.

- **Using classes might be slower** if you have a lot of small ones, which brings us to the next point.

- **Keep data that is used together, close together.** Closeness can be in either in space or time, or both. (Your CPU has multiple types of memory (from large to small, and slow to fast): HDD/SSD, RAM, L3, L2, L1 and registers. Data closeness will mean data needed is in a memory further to the right (closer to data just used, which is in the registers), therefore accessed more quickly). **Memory access speed is very often the ultimate bottleneck, not CPU speed.**

### Documentation

- **Types are a form of documentation.** You can tell what types your variables are, including function arguments, and what type a function returns. In Python these are called type hints, they are explained in PEP484. You can use the `mypy` tool to check whether the types used in your program are consistent. This might save you only a little time and effort for smaller programs, but is crucial for larger codebases.

### Structure

This section mixes structure for the entire program with the structure in functions.

- **`while` loops are almost never necessary.** Use `for` loops.

- **If possible, put `if` and `switch` statements outside of loops.** The CPU makes a guess every time it sees an `if`; when wrong it takes a lot of time to correct. Having a lot of hard to guess `if` statements will make these decisions often wrong and you will be left waiting.

- **Use as little indentation as possible, no more than three levels.**

- **It's okay to just start programming**, but at some point you will need to structure your code.

- **A simulation consists of the steps** initialisation, equilibration, and simulation—and possibly deconstruction afterwards.

- **We've found that making the simulation a class**, with initialisation in the constructor, and functions `equilibrate` and `simulate` works very well. A class is also ideal for storing the current state of the simulation.

### Optimisation

Some random statements about optimisation.

**The biggest takeaway is that a working simple program always trumps a non-working complex one.**

**Most importantly: Make something work, then make it readable and documented, then make it fast. In that order.**

- **Don't try to be smart**, unless it takes a huge amount of time to compute something in a less smart way.

- **Use a profiler to see which part/function uses the most time**, optimise that first. For Python, add `-m cProfile` to the Python command. Alternatively, consider the `pyspy`.

- **Test to see whether what you think would be faster is actually faster** using a profiler.

- **The interpreter/compiler is a lot smarter than you think.**

### Python-specific

- **Python has a package manager called `pip`.**

- **`numpy` is your friend**, it has its own arrays, and many convenient functions, from random sampling to linear algebra. A lot of smart people have put a lot of time and effort in making efficient versions of the functions. It is unlikely any of us will outsmart them in the context of this course. Using the numpy-provided functions when applicable to your problem is hence highly recommended.

- **If not using numpy's anti-for-loop functionality**, use the construction `for particle in particles:` whenever possible. If you also need an index, use `enumerate`: `for i, vel in enumerate(velocities):`. If you need to loop over more than one thing together, use `zip`: `for pos, vel in zip(positions, velocities):`.

- **If you are limited by the speed of some loops**, consider using `numba`.

- **`matplotlib` or `seaborn` are recommended for plotting.**

- **`lmfit` is pretty good for fitting functions to other functions.**

- **Need a progress bar?** Check out the `tqdm` package.

- **Python is faster in functions.** Therefore, make a `main()` function and call it in the `if __name__ == "__main__":`.

- **Use static-type checking to your advantage** by using type hints (PEP484 and `mypy`)

- **Use `pylint` to see how your code can be made to conform to standard Python.** However, note that `pylint` is very opinionated, but defaults can be changed.
