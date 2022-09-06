<a href="https://oscar-defelice.github.io"><img src="https://user-images.githubusercontent.com/49638680/98257151-9f5e5800-1f7f-11eb-9f42-479a4fc6cf24.png" height="125" align="right" /></a>

# Deep Learning lectures

Here is the material for a course of two-weeks I will be giving in a Master of Data Science and AI

This is part of a series of other lectures modules on

1. [Introduction to Data Science](https://oscar-defelice.github.io/DSAcademy-lectures) 🧮
2. [Statistical Learning](https://oscar-defelice.github.io/ML-lectures) 📈
3. [Time Series](https://oscar-defelice.github.io/TimeSeries-lectures) ⌛
4. [Computer Vision Hands-On](https://oscar-defelice.github.io/Computer-Vision-Hands-on) 🕶️

---

<p align="center">
    <img width="699" alt="image" src="https://user-images.githubusercontent.com/49638680/159042792-8510fbd1-c4ac-4a48-8320-bc6c1a49cdae.png">
</p>

---

## [Content of lectures](https://oscar-defelice.github.io/DeepLearning-lectures/src)

You can find the list of the arguments and some relevant material [here](https://oscar-defelice.github.io/DeepLearning-lectures/src).

---

## Install requirements

As usual, it is advisable to create a virtual environment to isolate dependencies.
One can follow [this guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) and the suitable section according to the OS.

Once the virtual environment has been set up, one has to run the following instruction from a command line

```bash
pip install -r requirements.txt
```

This installs all the packages the code in this repository needs.

### Mac M1 processors

For new Apple M1 processors, there is a different requirement file.
So set the virtual environment and then the command to execute is

```bash
pip install -r requirements-macm1.txt
```

This installs all the packages the code in this repository needs.

## Interact with notebooks

### Binder

You can use _Binder_, to interact with notebooks and play with the code and the exercises.

<p align="center">
<a href = "https://mybinder.org/v2/gh/oscar-defelice/DeepLearning-lectures/HEAD?urlpath=lab"> <img src="https://mybinder.org/badge_logo.svg"> </a>
</p>

### DeepNote

Alternatively, you can work on these notebooks in another online workspace called [Deepnote](https://www.deepnote.com/). This allows you to play around with the code and access the assignments from your browser.
<p align="center">
  <a href = "https://beta.deepnote.com/launch?template=data-science&url=https%3A%2F%2Fgithub.com/oscar-defelice/DeepLearning-lectures"> <img height="20" src="https://deepnote.com/buttons/launch-in-deepnote-small.svg"> </a>
</p>

## Run lectures in a Docker container

<p align="center">
  <img height="170" src="https://korben.info/app/uploads/2020/06/docker.png">
</p>

Another option to run all these lectures locally is to build the corresponding Docker Image.
A nice introduction to Docker containers can be found [here](https://www.youtube.com/watch?v=JprTjTViaEA).

We tried to modularise everything to make all the building and execution procedure as simple as possible.
To run a jupyter environment with all dependencies installed and notebooks ready to be executed it is sufficient to open your favourite terminal and run

```bash
make
```

The Makefile will take care of building and executing the docker image.
Then a jupyter server will be running at [http://localhost/8888](http://localhost/8888).

---

## Your lecturer 👨‍🏫
### [Oscar de Felice](https://oscar-defelice.github.io/)

<a href="https://oscar-defelice.github.io/" target="_blank" rel="that's me!">![Oscar](https://oscar-defelice.github.io/images/OscarAboutMe.png)</a>

I am a theoretical physicist, a passionate programmer and an AI curious.

I write medium articles (with very little amount of regularity), you can read them [here](https://oscar-defelice.medium.com/).
I also have a [github profile](https://github.com/oscar-defelice) where I store my personal open-source projects.

📫 [Reach me!](mailto:oscar.defelice@gmail.com)

[![github](https://img.shields.io/badge/GitHub-100000?style=plastic&logo=github&logoColor=white)](https://github.com/oscar-defelice)
[![Website](https://img.shields.io/badge/oscar--defelice-oscar-orange?style=plastic&logo=netlify&logoColor=informational&link=oscar-defelice.github.io)](https://oscar-defelice.github.io)
[![Twitter Badge](https://img.shields.io/badge/-@OscardeFelice-1ca0f1?style=plastic&labelColor=1ca0f1&logo=twitter&logoColor=white&link=https://twitter.com/oscardefelice)](https://twitter.com/OscardeFelice)
[![Linkedin Badge](https://img.shields.io/badge/-oscardefelice-blue?style=plastic&logo=Linkedin&logoColor=white&link=https://linkedin.com/in/oscar-de-felice-5ab72383/)](https://linkedin.com/in/oscar-de-felice-5ab72383/)

#### Questions

<p align="center">
  <img width="1269" alt="image" src="https://user-images.githubusercontent.com/49638680/167115562-1a780ea9-16d4-408b-a500-cd6ad740983d.png">
</p>

If you have any question, doubt or if you find mistakes, please open an issue or drop me an [email](mailto:oscar.defelice@gmail.com).


#### Buy me a coffee ☕️

If you like these lectures, consider to buy [me a coffee ☕️ ](https://github.com/sponsors/oscar-defelice)!

<p align="center">
  <a href="https://github.com/sponsors/oscar-defelice"><img src="https://raw.githubusercontent.com/oscar-defelice/DeepLearning-lectures/master/src/images/breakfast.gif"></a>
</p>

---

<p align="left">
<a href = "https://hub.docker.com/repository/docker/oscardefelice/deep-learning-lectures/general"> <img src="https://img.shields.io/docker/automated/oscardefelice/deep-learning-lectures?style=social"> </a>&nbsp;
<a href = "https://github.com/oscar-defelice/DeepLearning-lectures"> <img src="https://img.shields.io/github/stars/oscar-defelice/DeepLearning-lectures?style=social"> </a>&nbsp;
<a href = "https://oscar-defelice.github.io/DeepLearning-lectures"> <img src="https://img.shields.io/badge/website-up-informational?style=social"> </a>&nbsp;
</p>
