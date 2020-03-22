FROM    python:3.6.10-slim

WORKDIR /usr/src/app/

COPY    requirements.txt ./
RUN     pip install --no-cache-dir -r requirements.txt

COPY    . .

RUN     mkdir www && mv perceptron.html www/

ENTRYPOINT [ "python", "./perceptron.py" ]
CMD     [""]