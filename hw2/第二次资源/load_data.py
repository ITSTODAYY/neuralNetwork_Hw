import numpy as np
import matplotlib.pyplot as plt

mlp = []
cnn = []
for i in range(1,7):
    mlp.append(dict(np.load("resultMLP"+str(i)+".npy").tolist()))
    cnn.append(dict(np.load("resultCNN"+str(i)+".npy").tolist()))


plt.figure()
plt.subplot(221)
plt.title("mlp loss")
plt.plot(mlp[0]["trainl"], color = "yellow", label = "0.9 ")
plt.plot(mlp[0]["validl"], linestyle = '--', color = "yellow")
plt.plot(mlp[1]["trainl"], color = "red", label = "0.7 ")
plt.plot(mlp[1]["validl"], linestyle = '--', color = "red")
plt.plot(mlp[2]["trainl"], color = "blue", label = "0.5 ")
plt.plot(mlp[2]["validl"], linestyle = '--', color = "blue")
plt.plot(mlp[3]["trainl"], color = "green", label = "0.3 ")
plt.plot(mlp[3]["validl"], linestyle = '--', color = "green")
plt.plot(mlp[4]["trainl"], color = "purple", label = "0.1 ")
plt.plot(mlp[4]["validl"], linestyle = '--', color = "purple")
plt.plot(mlp[5]["trainl"], color = "orange", label = "0 ")
plt.plot(mlp[5]["validl"], linestyle = '--', color = "orange")

plt.legend()
#plt.xlabel("iteration")
plt.ylabel("loss value") 

plt.subplot(222)
plt.title("mlp acc")
plt.plot(mlp[0]["traina"], color = "yellow", label = "0.9 ")
plt.plot(mlp[0]["valida"], linestyle = '--', color = "yellow")
plt.plot(mlp[1]["traina"], color = "red", label = "0.7 ")
plt.plot(mlp[1]["valida"], linestyle = '--', color = "red")
plt.plot(mlp[2]["traina"], color = "blue", label = "0.5 ")
plt.plot(mlp[2]["valida"], linestyle = '--', color = "blue")
plt.plot(mlp[3]["traina"], color = "green", label = "0.3 ")
plt.plot(mlp[3]["valida"], linestyle = '--', color = "green")
plt.plot(mlp[4]["traina"], color = "purple", label = "0.1 ")
plt.plot(mlp[4]["valida"], linestyle = '--', color = "purple")
plt.plot(mlp[5]["traina"], color = "orange", label = "0 ")
plt.plot(mlp[5]["valida"], linestyle = '--', color = "orange")
plt.legend()
#plt.xlabel("iteration")
plt.ylabel("accuracy rate") 

plt.subplot(223)
plt.title("cnn loss")
plt.plot(cnn[0]["trainl"], color = "yellow", label = "0.9 ")
plt.plot(cnn[0]["validl"], linestyle = '--', color = "yellow")
plt.plot(cnn[1]["trainl"], color = "red", label = "0.7 ")
plt.plot(cnn[1]["validl"], linestyle = '--', color = "red")
plt.plot(cnn[2]["trainl"], color = "blue", label = "0.5 ")
plt.plot(cnn[2]["validl"], linestyle = '--', color = "blue")
plt.plot(cnn[3]["trainl"], color = "green", label = "0.3 ")
plt.plot(cnn[3]["validl"], linestyle = '--', color = "green")
plt.plot(cnn[4]["trainl"], color = "purple", label = "0.1 ")
plt.plot(cnn[4]["validl"], linestyle = '--', color = "purple")
plt.plot(cnn[5]["trainl"], color = "orange", label = "0 ")
plt.plot(cnn[5]["validl"], linestyle = '--', color = "orange")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("loss value") 

plt.subplot(224)
plt.title("cnn acc")
plt.plot(cnn[0]["traina"], color = "yellow", label = "0.9 ")
plt.plot(cnn[0]["valida"], linestyle = '--', color = "yellow")
plt.plot(cnn[1]["traina"], color = "red", label = "0.7 ")
plt.plot(cnn[1]["valida"], linestyle = '--', color = "red")
plt.plot(cnn[2]["traina"], color = "blue", label = "0.5 ")
plt.plot(cnn[2]["valida"], linestyle = '--', color = "blue")
plt.plot(cnn[3]["traina"], color = "green", label = "0.3 ")
plt.plot(cnn[3]["valida"], linestyle = '--', color = "green")
plt.plot(cnn[4]["traina"], color = "purple", label = "0.1 ")
plt.plot(cnn[4]["valida"], linestyle = '--', color = "purple")
plt.plot(cnn[5]["traina"], color = "orange", label = "0 ")
plt.plot(cnn[5]["valida"], linestyle = '--', color = "orange")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("accuracy rate") 


plt.show()

maxnum = -1000
for i in range (0,6):
    maxnum = -1000
    for num in mlp[i]["traina"]:
        if num > maxnum:
            maxnum = num
    print(maxnum)
    print("\n")

    maxnum = -1000
    for num in mlp[i]["valida"]:
        if num > maxnum:
            maxnum = num
    print(maxnum)
    print("\n")

    maxnum = -1000
    for num in cnn[i]["traina"]:
        if num > maxnum:
            maxnum = num
    print(maxnum)
    print("\n")

    maxnum = -1000
    for num in cnn[i]["valida"]:
        if num > maxnum:
            maxnum = num
    print(maxnum)
    print("\n")


    
