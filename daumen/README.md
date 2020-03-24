 # Dockerfile for Data AUgMENtation
 ## About
 This Dockerfile will install all the necessary dependencies for the Data AUgMENtation project and also build it so that `daumen` works out of the box, in particular for using the Page Segmentation tool. The [Data AUgMENtation project](https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen) was developed by Cristoph Wick et al..
 ## Building the Docker image
 Run `docker build -t daumen .` to build the docker image. You can replace daumen with another name if you like.
 ## Accessing the terminal
 Run `docker run -ai daumen` to access the bash terminal inside the newly created Docker image.
 ## Re-accessing the terminal after exit
 Running `docker run` will create a new container for the Docker image. This container should continue to exist even after you shutdown your computer. To re-access the terminal inside the Docker image using this container, first figure out the container ID using `docker ps -a`, and then run `docker start -ai CONTAINER_ID` (replacing CONTAINER_ID with the continer ID you found using `docker ps`). You can also just use the container name in place of the container ID.
 ## Clearing up space
 Should you need to delete the Docker image to save some space on your host machine, first delete the container using `docker container rm CONTAINER_ID` (you can also just use the container name here) and then run `docker image rm daumen` (or however you named your image).
