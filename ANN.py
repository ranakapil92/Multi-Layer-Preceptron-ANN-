import pandas as pd                                         #for arranging data into matricex
from csv import reader                                      #for reading the fiile    
from pandas import DataFrame                                #for data arrangement
import matplotlib.pyplot as plt                             #for ploting the graph
import numpy as np                                          #for matrix multipication  

                                 
def mean(attr):
    return sum(attr) / float(len(attr))        

def variance(attr, mean):
	return sum([(x-mean)**2 for x in attr])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

inputdata = list()
with open("data.txt", 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        inputdata.append(row)

rows= len(inputdata)


for i in range(0,rows):
        inputdata[i]=[1.0]+inputdata[i]          #adding bias feature to input set

        
        
dataset = DataFrame(inputdata)
  
        
for i in range(0,401):
    dataset[i]= dataset[i].astype(float) 
      
        
dataset.dtypes        
        
X_main = dataset.iloc[:,:401].values
    
inputdata = list()
with open("label.txt", 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        inputdata.append(row)

rows= len(inputdata)        
        
dataset = DataFrame(inputdata)
        
for i in range(0,10):
    dataset[i]= dataset[i].astype(float) 
      
        
dataset.dtypes        
        
y_main = dataset.iloc[:,:10].values  

#Data Is Ready Till here      
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_main, y_main, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


D_o_out=np.zeros(10)
D_i_out=np.zeros(10)
D_o_hid=np.zeros(501)
D_i_hid=np.zeros(500)

results=np.zeros((100,10))  #This is for storing all error wrt to epochs.


for trial in range(0,5):
    
    W = np.random.normal(0, 1, (500,401))        #intialization of weights 
    V = np.random.normal(0, 1, (10,501))
    
    for epoch in range(0,100):
        
        W_grad=np.zeros((500,401))
        V_grad=np.zeros((10,501))
        W_grad_total=np.zeros((500,401))
        V_grad_total=np.zeros((10,501))

        total_training_error=0
        
        for i in range(0,len(X_train)):
            
           
            H_in=np.dot(W,X_train[i,:])            #input to first hidden layer
            H_out1=np.tanh(H_in)                   #output of first hidden layer 
           
            
            H_out = np.insert(H_out1, 0,1.0, axis=0)          #adding bias to hidden layer
            
          
            O_in=np.dot(V,H_out)                             #input to output layer
            
            O_soft=softmax(O_in)                            #output function , softmax
            label = y_train[i]                                  #actual labels
            
            error=np.zeros(10)
            D_out_temp=np.zeros(10)
            
            for j in range(0,10):
                if (label[j]!=0):
                    error[j]=0.0 - np.log(O_soft[j])                #calculating error for 1 instance
                    total_training_error=total_training_error + error[j]     #summing up errors
                    D_o_out[j]=0.0 - (1/O_soft[j])
                    D_i_out[j]=1-O_soft[j]
                    #print(error[j],O_soft)
                    
                else:
                    error[j]=0.0
                    D_o_out[j]=0.0
                    D_i_out[j]=(-1)*O_soft[j]
            
                    
            
            D_i_out=np.subtract(O_soft,label)                 #gradient of error wrt to output layer
            
# here we have derivative with respect to output layer     
# Backpropagation
            
            
            D_o_hid=np.dot(np.transpose(V),D_i_out)           #gradient of error wrt to output part of hidden layer
            
            V_grad=np.outer(D_i_out,H_out)                  #gradient of error wrt to weigts between hidden and output layer , V
            
            V_grad_total=V_grad_total + V_grad
            
            if((i+1)% 25 ==0):
                
                V_grad= 0.01*V_grad_total
                V=np.subtract(V,V_grad)                     # weight V updation in 25 batchs
                V_grad_total=np.zeros((10,501))
            
            D_temp=np.square(H_out1) 
            D_temp=1-D_temp
           
            
            D_i_hid=D_o_hid[1:501] * D_temp                 #gradient of error wrt to hidden ;ayer
            
            
        
            W_grad=np.outer(D_i_hid,X_train[i])            #gradient wrt to weigts between input and hidden layer , W
            
            W_grad_total=np.add(W_grad_total , W_grad)
            
            if((i+1)% 25 ==0):
                W_grad= 0.01*W_grad_total
                W=np.subtract(W,W_grad)                 # weight W updation in 25 batchs
                W_grad_total=np.zeros((500,401))
         
        total_training_error = total_training_error / len(X_train)    
        print(total_training_error)            
        results[epoch,(trial*2)]=total_training_error
        
   
# here we are computing error over validation set for each epoch
        
        
        total_val_error=0      
        for i in range(0,len(X_val)):
                    
            H_in=np.dot(W,X_val[i,:])
            H_out1=np.tanh(H_in)
               
            
            H_out = np.insert(H_out1, 0,1.0, axis=0)
            
            label = y_val[i]  
            O_in=np.dot(V,H_out)
            
            O_soft=softmax(O_in)     
            
            
            for j in range(0,10):
                  if (label[j]!=0):               
                      #print(label[j],O_soft[j])
                      error[j]=0.0 - np.log(O_soft[j])
                      total_val_error=total_val_error + error[j]
             
        total_val_error=total_val_error/(len(X_val)) 
        print(total_val_error)
        results[epoch,(trial*2)+1]=total_val_error
          
            
        
#plotting of graphs    

get_ipython().run_line_magic('matplotlib', 'qt')
    

xg2=np.arange(1,101)
plt.figure(" Training Error Vs Epoch")
for i in range(0,5):
    yg1=results[:,i*2]
    plt.plot(xg2, yg1, 'o-')
    plt.legend(['Training Error'], loc='upper right')
    plt.ylabel("Training Error")
    plt.xlabel("No Of Epochs")
    plt.rc('axes', labelsize=20)
    
 



mean_training_error=np.zeros(100)
mean_validation_error=np.zeros(100)
variance_training_error=np.zeros(100)
for i in range(0,100):
    
    a=results[i,:]
    a1 = a[::2]
    a2= a[1:][::2]
    mean_training_error[i] = sum(a1)/5.0
    a1=np.subtract(a1,mean_training_error[i])
    var=sum(np.square(a1))/5.0
    variance_training_error[i]=var
    mean_validation_error[i] = sum(a2)/5.0


    

xg2=np.arange(1,101)
plt.figure("Means Average Training and Validation Error vs Epoch")
yg1=mean_training_error[:]
yg2=mean_validation_error[:]
plt.plot(xg2, yg1, 'o-')
plt.plot(xg2, yg2, 'o-')
plt.legend(['Training Error', 'Validation Error'], loc='upper right')
plt.ylabel("Training Error/Validation Error")
plt.xlabel("No Of Epochs")
plt.rc('axes', labelsize=15) 
   


plt.figure("Mean,Variance Training Error vs Epoch")
x = np.arange(1,101)
var = variance_training_error[:]
y = mean_training_error[:]
plt.ylabel("Mean Training Error")
plt.xlabel("No Of Epochs")
plt.errorbar(x, y, yerr=var, fmt='.k');  


plt.figure("Mean,Variance Training Error (*1000) vs Epoch")
x = np.arange(1,101)
var = variance_training_error[:]*1000
y = mean_training_error[:]
plt.ylabel("Mean Training Error")
plt.xlabel("No Of Epochs")
plt.errorbar(x, y, yerr=var, fmt='.k');
plt.show()  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        