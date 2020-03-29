import os
import sys
import getopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import http.server
import socketserver
import threading
import signal


class Perceptron:
    def __init__(self, cvsFile, seed=42, targetIMGPath='.'):
        self.seed = seed
        self.learn_rate = 0.01
        self.num_epochs = 25
        self.targetIMGPath = targetIMGPath

        data = pd.read_csv(cvsFile, encoding='utf-8')
        self.redPoints = pd.DataFrame()
        self.bluePoints = pd.DataFrame()
        self.X = data.drop('class', axis=1)
        self.y = data['class']

        for index, row in data.iterrows() :
            if(row['class'] == 1.0) :
                self.bluePoints = self.bluePoints.append(data.iloc[index])
            else :
                self.redPoints  = self.redPoints.append(data.iloc[index])

        # Setting the random seed, feel free to change it and see different solutions.
        np.random.seed(seed)


    def graph(self, lines) :
        plt.scatter(self.redPoints['x'], self.redPoints['y'], color='red')
        plt.scatter(self.bluePoints['x'], self.bluePoints['y'], color='blue')
        plt.axis([-0.5, 1.5, -0.5, 1.5])

        for i in range(len(lines)) :
            plt.plot([lines[i][1][0], lines[i][0][0]+lines[i][1][0]], 'g--', linewidth=1)

        plt.xticks()
        plt.yticks()

        #plt.show()
        plt.savefig(self.targetIMGPath + '/perceptron_graph.png')


    def stepFunction(self, t):
        if t >= 0:
            return 1
        return 0


    def prediction(self, X, W, b) :
        return self.stepFunction((np.matmul(X,W)+b)[0])


    def perceptronStep(self, X, y, W, b, learn_rate = 0.01) :
        for i in range(len(X)):
            y_hat = self.prediction(X.iloc[i],W,b)
            if (y[i]-y_hat == 1) : 
                    W[0] = W[0] + (X.iloc[i].T[0]*learn_rate)
                    W[1] = W[1] + (X.iloc[i].T[1]*learn_rate)
                    b += learn_rate
            elif (y[i]-y_hat == -1) :
                    W[0] = W[0] - (X.iloc[i].T[0]*learn_rate)
                    W[1] = W[1] - (X.iloc[i].T[1]*learn_rate)
                    b -= learn_rate
        return W, b


    def trainPerceptronAlgorithm(self, learn_rate = None, num_epochs = None):
        if learn_rate is None:
            learn_rate = self.learn_rate
        if num_epochs is None:
            num_epochs = self.num_epochs
        X = self.X
        y = self.y
        W = np.array(np.random.rand(2,1))
        b = np.random.rand(1)[0] + max(X.T[0])

        # These are the solution lines that get plotted below.
        boundary_lines = []
        for i in range(num_epochs) :
            # In each epoch, we apply the perceptron step.
            W, b = self.perceptronStep(X, y, W, b, learn_rate)
            boundary_lines.append((-W[0]/W[1], -b/W[1]))
            # remove 'Unsued variable 'i' warning
            i=i

        self.graph(boundary_lines)
        return


def signal_handler(sig, frame):
    ''' Signal handler to catch Ctrl-l signal from
        keyboard. Does nothing but to stop the main
        thread flow until this signal is catched. '''
    print("Catched signal Ctrl-l.")
    print("Good bye.")


def runServer(serv_port, server):
    ''' Calls the 'serve_forever' method of the server
        given as parameter. Intention here is to run 
        the server in a different thread than the main
        thread. '''
    print("serving at port: ", serv_port)
    print("Press Ctrl-l to stop the socket server...")
    server.serve_forever()

def main(argv):
    # by default, we take our own dataframe and parameters
    cvs_file = './data.csv'
    learn_rate = 0.01
    num_epochs = 25
    seed = 42
    # server port will be ....
    serv_port = 8080

    # check that the given inputs comply with the input
    # arguments requirements
    try:
        opts, args = getopt.getopt(argv, "i:l:e:s:p:",
                        ["input-data=","learning-rate=","num-epochs=","seed=", "tcp-server-port="])
                        
    except getopt.GetoptError:
        print("Incorrect ussage of arguments.")
        print("Correct ussage is:")
        print("     perceptron.py [-i <input_cvsfile>][-l <float learning rate>][-e <int number of epochs][-s <int seed>][-p <int port tcp server]")
        print("You can also substitute the flags as follows:")
        print("     --input-data        vs. -i")
        print("     --learning-rate     vs. -l")
        print("     --num-epochs        vs. -e")
        print("     --seed              vs. -s")
        print("     --tcp-server-port   vs. -p")
        sys.exit(2)

    # process the -possible- given arguments
    for opt, arg in opts:
        if opt in ("-i", "--input-data"):
            cvs_file = arg
            if not cvs_file.endswith('.csv'):
                print("Input file must be a csv file with three columns (i.e. x, y, class). Ej.")
                print("x,y,class")
                print("0.78051,-0.063669,1")
                print("0.28774,0.29139,1")
                print("0.40714,0.17878,1")
                print("......")
                sys.exit(2)
        elif opt in ("-l", "--learning-rate"):
            learn_rate = float(arg)
        elif opt in ("-e", "--num-epochs"):
            num_epochs = int(arg)
        elif opt in ("-s", "--seed"):
            seed = int(arg)
        elif opt in ("-p", "--tcp-server-port"):
            serv_port = int(arg)
    # remove warning "Unsued variable 'args'"
    args=args

    # instantiate from the Perceptron class and train it -
    # (according to the given mandatory parameters)
    perceptron = Perceptron(cvs_file, seed=seed, targetIMGPath='www')
    perceptron.trainPerceptronAlgorithm(learn_rate, num_epochs)

    # change to the www directory to start the server from there
    # (i.e. to serve that directory only)
    web_dir = os.path.join(os.path.dirname(__file__), 'www')
    os.chdir(web_dir)

    handler = http.server.SimpleHTTPRequestHandler
    server = socketserver.TCPServer(("", serv_port),handler)

    runningServer_thread = threading.Thread(
        name="RunningServer_thr", target=runServer, args=(serv_port,server,), daemon=False
    )
    runningServer_thread.start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    server.shutdown()


if __name__=="__main__":
    main(sys.argv[1:])