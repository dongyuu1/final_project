import csv
import matplotlib.pyplot as plt

csv_reader = csv.reader(open("log.csv"))
epochs = []
test_loss = []
test_accu = []
test_mIOU = []

for row in csv_reader:
  epochs.append(int(row[0][7:]))
  test_loss.append(float(row[1][11:]))
  test_accu.append(float(row[2][11:]))
  test_mIOU.append(float(row[-1][16:]))


plt.plot(epochs[:60], test_loss[:60], label="parent")
plt.plot(epochs[60:120], test_loss[60:120], label="nearest")
plt.plot(epochs[120:], test_loss[120:], label="octformer")
plt.xlabel("epochs")
plt.ylabel("test loss")
plt.legend()
plt.savefig('./visualisation/test_loss.png')
plt.show()


plt.plot(epochs[:60], test_accu[:60], label="parent")
plt.plot(epochs[60:120], test_accu[60:120], label="nearest")
plt.plot(epochs[120:], test_accu[120:], label="octformer")
plt.xlabel("epochs")
plt.ylabel("test OA")
plt.legend()
plt.savefig('./visualisation/test_OA.png')
plt.show()


plt.plot(epochs[:60], test_mIOU[:60], label="parent")
plt.plot(epochs[60:120], test_mIOU[60:120], label="nearest")
plt.plot(epochs[120:], test_mIOU[120:], label="octformer")
plt.xlabel("epochs")
plt.ylabel("test mIOU")
plt.legend()
plt.savefig('./visualisation/test_mIOU.png')
plt.show()
