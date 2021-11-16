FROM continuumio/miniconda3


WORKDIR /analysis


# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment.yml .
COPY README.md .
RUN conda env create -f environment.yml

# Initialize conda in bash config fiiles:
RUN conda init bash
# VOLUME "data"

# Activate the environment, and make sure it's activated:
RUN echo "conda activate info529midterm" > ~/.bashrc
RUN echo "Make sure joblib is installed:"
RUN python -c "import joblib; print('joblib imported nice')"

RUN echo "Beginning to add relevant directories to container"

# The code to run when container is started:
ADD scripts/ scripts/
ADD results/ results/
RUN mkdir data
 
WORKDIR /analysis/scripts
RUN pwd
RUN ls -l
RUN chmod +x ensemble_train_bash_dev.sh

RUN apt-get -y update

RUN apt-get -y install unzip

RUN echo "Beginning Ensemble Below"
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "info529midterm", "/analysis/scripts/ensemble_train_bash_dev.sh"]

# ENTRYPOINT ["/analysis/scripts/ensemble_train_bash_dev.sh"]
