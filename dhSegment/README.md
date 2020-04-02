 ## Building the Docker image
 Run `docker build -t dhseg .` to build the docker image. You can replace "daumen" with another name if you like.
 ## Accessing the terminal
 Run `docker run -ti dhseg ` (or however you named your image) to access the bash terminal inside the newly created Docker image.

 ## Exiting the terminal 
 press ctrl+d
 Right now, when you exit the terminal the instance of your container gets deleted and the progress you made goes along with it. I am currently trying to find a way to make it so that the image pauses when you exit the terminal.  

 ## *To Grab the Pretrained Page Segmentation and BaseLine Models* 
 wget https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip

 rm -r /content/model;unzip model.zip

 wget https://github.com/dhlab-epfl/fdh-tutorials/releases/download/v0.1/line_model.zip

 rm -r /content/polylines;rm -r /content/__MACOSX;unzip line_model.zip

 (The page segmetation model is required for the baseline model to work properly)

## *Grab the Provided Demo Images*
wget https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip

unzip pages.zip

## *Grab READ-BAD labeled ICDAR 2017 Baseline  Detection Dataset (Has bounding boxes)*
wget https://zenodo.org/record/1491441/files/READ-ICDAR2017-cBAD-dataset-v4.zip?download=1

unzip READ-ICDAR2017-cBAD-dataset-v4.zip?download=1

(Download will take an obnoxiously long time)
