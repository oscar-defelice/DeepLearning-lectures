<p align="center">
    <img width="512" alt="jupyterlogo" src="https://user-images.githubusercontent.com/49638680/203304584-729686ab-eb60-4641-8b4f-1a0e5a91ed4d.png">
</p>

# Jupyter Notebook cheatsheet

In this article, I will walk you through some simple tricks on how to improve your experience with Jupyter Notebook. We will start from useful shortcuts and we will end up adding themes, automatically generated table of contents.

## Introduction

As you know, Jupyter Notebook is a client-server application used for running notebook documents in the browser. Notebook documents are documents able to contain both code and rich text elements such as paragraphs, equations, etc...

Jupyter Notebook is nowadays probably the most used environment for solving Machine Learning/Data Science tasks in Python.
Although it is a good debug and experiment environment and allows many nice things from the point of view of visualisation (code documentation, inline graphs, etc.), you have to keep in mind, Jupyter Notebook is __not__ a development environment. 

The main reason for this is the fact that notebooks have tons and tons of hidden state that is easy to screw up and difficult to reason about. This makes notebook really difficult to debug and to version efficiently.

It is true that nowadays there are tools out there (_e.g._ [Ploomber](https://blog.jupyter.org/ploomber-maintainable-and-collaborative-pipelines-in-jupyter-acb3ad2101a7?gi=ed373e9ae21a)) able to make you use your notebook to build pipelines and put those in production, however, my feeling is that these great tools are more a workaround such that people do not have to fix their bad habits.

Just to have a reference, there is [this nice presentation](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g38857eff70_0_0) at the Jupyter Conference 2018 that collects a lot of well known Jupyter notebooks environment problems.

However, since these lectures are _demanded_ to be on jupyter notebooks, let's start by getting the max out of it.

## Commandments

1. You will always use notebook responsibly, _i.e._ you never execute a notebook you do not understand jus tto get to the end.
2. You will always run cells in order, if you need to rerun a previous cell, _always_ restart the kernel.
3. You will always prefer jupyterlab to jupyter-notebook.

## Shortcuts

Shortcuts can be really useful to speed up writing and executing your code, I will now walk you through some of the shortcuts I found most useful to use in Jupyter.

There are two possible way to interact with Jupyter Notebook: __Command Mode__ and __Edit Mode__.

Some shortcuts works only on one mode or another while others are common to both modes.

### Common shortcuts

Some shortcuts which are common in both modes are:

* __Ctrl + Enter__ : to run all the selected cells;
* __Shift + Enter__ : run the current cell and move the next one;
* __Ctrl + s__ : save notebook.

### Command mode shortcuts

In order to enter Jupyter command mode, we need to press _Esc_ and then any of the following commands, the cell selection will change colour (which one depends on you theme):

* __H__ : show all the shortcuts available in Jupyter Notebook
* __Shift + Up/Down Arrow__ : to select multiple notebook cells at the same time (pressing enter after selecting multiple cells, will make run all of them!);
* __A__ : insert a new cell above;
* __B__ : insert a new cell below;
* __X__ : cut the selected cells;
* __Z__ : undo the deletion of a cell;
* __Y__ : change the type of cell to Code;
* __M__ : change the type of cell to Markdown;
* __Space__ : scroll notebook down;
* __Shift + Space__ : scroll notebook up.

### Edit mode shortcuts

In order to instead enter Jupyter edit mode, we need to press Enter and successively any of the following commands:

* __Tab__ : code completition suggestions;
* __Ctrl + ]__ : indent code;
* __Ctrl + [__ : dedent code;
* __Ctrl + z__ : undo;
* __Ctrl + y__ : redo;
* __Ctrl + a__ : select all;
* __Ctrl + Home__ : move cursor to cell start;
* __Ctrl + End__ : move cursor to the end of the cell;
* __Ctrl + Left__ : move cursor one word left;
* __Ctrl + Right__ : move cursor one word right.

## Jupyter Themes

If you are interested in changing how your Jupyter notebook looks like, it is possible to install a package with a collection of different themes. The default Jupyter theme looks like the one in Figure $1$, in Figure $2$ you will see how we will be able to personalise its aspect.

<p align="center">
    <img alt="Figure1" width="400" src="https://www.freecodecamp.org/news/content/images/2019/08/2-2.PNG"> <img alt="Figure2" width="400" src="https://www.freecodecamp.org/news/content/images/2019/08/1-1.PNG"> 
</p>

You can install a theme package by running in a shell (or in a magic cell in a terminal)

```bash
pip install jupyterthemes
```

and to see the themes available,

```bash
jt -l
```

Finally, we can choose a theme using the following command (in this example I decided to use the `solarizedl` theme):

```bash
jt -t solarizedl
```

If you want to come back to your original theme, it is sufficient to run,

```bash
jt -r
```

## Jupyter Notebook Extensions

Notebook extensions can be used to enhance user experience offering a wide variety of personalizations techniques.

In this example, I will be using the `nbextensions` library in order to install all the necessary widgets. Such library makes use of different Javascript models in order to enrich the notebook frontend.

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --system
```

Once `nbextensions` is installed you will notice that there is an extra tab on your Jupyter notebook homepage.

<p align="center">
    <img alt="Figure3" src="https://www.freecodecamp.org/news/content/images/2019/08/image-128.png"> 
</p>

By clicking on the _Nbextensions_ tab, we will be provided with a list of available widgets. 
In my case, I decided to enable the ones shown below.

<p align="center">
    <img alt="Figure4" src="https://www.freecodecamp.org/news/content/images/2019/08/image-129.png"> 
</p>

Some of my favourite extensions are:

> Table of Contents

Auto-generate a table of contents from markdown headings.

<p align="center">
    <img alt="toc" src="https://www.freecodecamp.org/news/content/images/2019/08/ezgif.com-optimize-1.gif"> 
</p>

> Snippets

Sample codes to load common libraries and create sample plots which you can use as starting point for your data analysis 

<p align="center">
    <img alt="toc" src="https://www.freecodecamp.org/news/content/images/2019/08/snippets.gif"> 
</p>

> Hinterland

Code autocompletion for Jupyter Notebooks.

<p align="center">
    <img alt="toc" src="https://www.freecodecamp.org/news/content/images/2019/08/completition.gif"> 
</p>

The nbextensions library provides many other extensions apart for these mentioned here, I encourage you to experiment and test any-other which can be of interest for you.

## Markdown Options

By default, the last output in a Jupyter Notebook cell is the only one that gets printed (There is an implicit `display()` command).

Additionally, it is possible to write LaTex in a Markdown cell by enclosing the text between dollar signs ($).

## Notebook Slides

It is possible to create a slideshow presentation of a Jupyter Notebook by going to `View -> Cell Toolbar -> Slideshow` and then selecting the slides configuration for each cell in the notebook.

Finally, going to the terminal and typing the following commands the slideshow will be created.

```bash
pip install jupyter_contrib_nbextensions
jupyter nbconvert my_notebook_name.ipynb --to slides --post serve
```

<p align="center">
    <img alt="toc" src="https://www.freecodecamp.org/news/content/images/2019/08/ezgif.com-optimize--1-.gif"> 
</p>

## Cell Magic

Magics are commands which can be used to perform specific commands. Some examples are: inline plotting, printing the execution time of a cell, printing the memory consumption of running a cell, etc…

Magic commands which starts with just one % apply their functionality just for one single line of a cell (where the command is placed). Magic commands which instead starts with two %% are applied to the whole cell.

It is possible to print out all the available magic commands by using the following command,

```python
%lsmagic
```

