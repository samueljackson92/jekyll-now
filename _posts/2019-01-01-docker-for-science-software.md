---
layout: post
title: Docker for Scientific Software
date: 2019-01-01
tags: docker, software, opinion
published: False
---

*A colleague asked me the other day why I prefer Docker over Conda. Here's my thoughts on a response.*

In my work I frequently work with different scientific programs, libraries, tools, etc. created by the community. While it is awesome that many of these pakcages are free and open source, I often find that the packaging of scientific software I want to install ranges from "not great" to "does this compile on anything but the creators machine?". How many times have you sat down to install a library you need and found you can't simply `pip install name-of-thing`? How many times have you had to delve into a Makefile so you can successfully compile a Fortran library that predates your existance?

People who write Python (or R) have typically turned to virtual envrionments and [Anaconda](https://www.anaconda.com/distribution/) to solve such problems. For common, fairly well known packages I would argue that this works brilliantly. It allows me to install things in a seperate environment and I can even install dependencies a specifc compiler version into the environment for software that has extension that need to be compiled using something that the host OS doesn't support. 

However, I tend to find that Anaconda only gets you so far. A couple of stumbling blocks I've encourntered more than once are:

- What if the package I want is not already on PyPI or the anaconda cloud? Or even has a `setup.py`?
- What if the software I need has some dependancy that isn't on Anaconda/PyPI?
- What if the software only compiles/runs on a different OS to the one I'm currently working on?
- What if I need to build the software from source?
  - Either because I have funky requirements or because there's no other way!
- What if once I've gone through all the effort of solving the problems above I want to share my setup with colleagues?

I argue that [Docker](https://www.docker.com/) solves most of these problems and provides some addtional benefits too. The article isn't intended to be a full docker tutorial. For that you can head over to the [docker site](https://docs.docker.com/get-started/) or check out one of the many excellent tutorials online. Here I'm just going to justify why you should be interested. 

![image](https://upload.wikimedia.org/wikipedia/commons/4/4e/Docker_%28container_engine%29_logo.svg)

Docker is a program that provides the ability to "containerize" applications and run them in an extremely thin virualisation envrionment ontop of your host machine. It's a bit like running a virtual machine, but instead of the whole OS you only bundle the applications/tools/libraries etc. that you need into the container and all containers share a common operating system kernel. If you've not already heard of docker or haven't yet had the chance to play around with it I strongly suggest investing the time because it's awesome.

Firstly, like Anaconda, the Docker community has [Docker Hub](https://hub.docker.com/) which provides a wide selection of pre built docker containers that users can install. Of particular interest to data scientists like myself are the [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/) which allow you to run a Jupyter lab envrionment from within a container.

Secondly, docker provides the ability to create new or extend exsiting docker images using an inheritence mechanism and a Dockerfile. An example of a Dockerfile is shown below. Basically you specify the parent image you want to inherit from `python:2.7-slim` in this case and then specify a bunch of commands to setup the environment as you require. The most important line to pay attention to in the line below is the `RUN` command which allows you to run any abritrary command on the base system (probably Ubuntu in the example below). 

This gives the user massive flexiability in how software should be installed/compiled within the envrionment. I can specifc the specific location things need to be installed in, I can install any arbitrary dependencies I need, I can compile things from scratch using any compiler of my choice etc. (not just limited to things in Anaconda for example).

```dockerfile
# Use an official Python runtime as a parent image
FROM python:2.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

*An example Dockerfile from the official docker tutorial.*

An added benefit is that you document how to software is installed into the environment *as you're installing stuff to it*. The Dockerfile setups the envrionment for you but also acts as a living document explicitly stating what's installed, where and how. So, if you ever need to install it on another system you have the list of commands that successfully got you there.

But this brings me to the next benefit of Docker. Once you've created and image of your envrionment you can share it on Docker hub. If someone else wants to run stuff using your painfully configured  envrionment, all they need to do is pull your image from Docker hub. This is also useful if I want to run the same envrionment on different machines, such as shareing my jupyter lab setup between my Mac at home and my Windows machine at work.

Another couple of benefits worth mentioning are the `docker-compose` feature and the NVIDIA docker support. `docker-compose` allows you spin up multiple docker instances at the same time. For example, I was recently working on a personal project where I wanted to run Jupyter lab instance and connect to a MongoDB instance. `docker-compose` allows you start and configure both containers with a single command. The specification of how to configure them is stored in a simple YAML file which can be shared along with the git repo for the code, making it super useful for others to mimick the setup.

[NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) is a project that aims to provide docker containers access to the hosts NIVIDA GPU. There's some simple overhead in installing the NVIDIA drivers to the host machine, but once that's done you can run your containers with a specific version of the NVIDIA toolkit you require which is great for testing & deploying GPU utilising software on different machines.

Hopefully I managed to express some of the benefits of docker and that potentially investing time in learning how to use docker can bring some productivity benefits.

