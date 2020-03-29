FROM    python:3.6.10-slim-stretch

WORKDIR /usr/src/app/

COPY    requirements.txt ./
RUN     pip install --no-cache-dir -r requirements.txt

COPY    . .

RUN     mkdir www && \
        mv perceptron.html www/index.html

ENTRYPOINT [ "python", "-u", "./perceptron.py" ]
CMD     [""]