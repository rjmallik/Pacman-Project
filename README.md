AI Agents Project

Hello!

This repository contains all material and  code for my personal exploration and development of Artificial Intelligence agents, built on top of foundational work from the UCSC's AI/ML course. While this is an independent project, I want to acknowledge and thank the course staff and contributors whose excellent materials served as a starting point for much of this work.

Credits:
A Python implementation of artificial intelligence search algorithms to solve problems within the Berkeley Pac-Man environment. The Pac-Man Projects, developed at UC Berkeley, apply AI concepts to the classic arcade game. I help Pac-Man find food, avoid ghosts, and maximize his game score using uninformed and informed state-space search, probabilistic inference, and reinforcement learning. Special thanks to the CSE 140 teaching team and the maintainers of the LINQS Pac-Man repository for providing an engaging framework and high-quality resources. This project builds on their foundations with extensions, modifications, and additional experiments conducted as part of my own learning and research into AI agent design.

Setup:
This project uses Python (version >= 3.8). If you're unfamiliar with Python, don't worry—this repository is also a learning tool and includes resources to help you get up to speed.

To set up the development environment:

- Ensure Python (>= 3.8) is installed.
- Set up a Python virtual environment.
- Install the required dependencies via requirements.txt.
- Make sure Tkinter is installed (used for visualizations).

The Pac-Man projects were developed for UC Berkeley's introductory artificial intelligence course, CS 188. Go here for more information/complete functionality for this project: http://ai.berkeley.edu/project_overview.html

Project Structure:
The project is divided into five parts, each building on the last to incrementally implement and test various AI strategies for controlling agents:

P0: Initial setup and environment configuration.

P1: Search algorithms for pathfinding.

P2: Multi-agent decision making.

P3: Probabilistic inference and tracking.

P4: Planning and learning in competitive environments (culminates in a Pac-Man agent tournament).

Autograder:
For automated evaluation, I use the open-source EduLinq Autograder, along with its Python interface, which is already listed as a requirement. This tool provides immediate feedback and ensures code correctness as I iterate through different AI implementations.
