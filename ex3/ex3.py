"""
Course Code and Title: SGN-41007 Pattern Recognition and Machine Learning
Exercise: 03
Author: Nahidul Islam
Student ID# 272487
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Question: 3(a)

n = np.arange(100)
f0 = 0.017
w = np.sqrt(0.25) * np.random.randn(100)
x = np.sin(2 * np.pi * f0 * n) + w
plt.figure(1)
plt.plot(n, x, 'bo')
plt.show()

# Question: 3(b)

scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    n = np.arange(100)
    z = -2 * np.pi * 1j * f * n  # <compute -2*pi*i*f*n. Imaginary unit is 1j>
    e = np.exp(z)
    score = np.abs(np.dot(e, x))  # <compute abs of dot product of x and e>
    scores.append(score)
    frequencies.append(f)

fHat = frequencies[np.argmax(scores)]
print(f"The value of fHat: {fHat}")

# Question: 3(c)

# Ans. Yes.

# Question : 4

digits = load_digits()
print(digits.keys())
plt.figure(2)
plt.gray()
plt.imshow(digits.images[0])
plt.show()
print(f"Label of the printed digit: {digits.target[0]}")
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.20)

# Question: 5

clf = KNeighborsClassifier()
model = clf.fit(x_train, y_train)
predicted_label = clf.predict(x_test)
accuracy = accuracy_score(predicted_label, y_test)
print("Accuracy: %.2f" % accuracy)
