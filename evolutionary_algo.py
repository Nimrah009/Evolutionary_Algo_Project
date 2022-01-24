# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EVOLUTIONARY ALGORITHUM FOR IMAGE RECOGNITION PROBLEM.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# You need to install python, matplotlib, numpy, random and cv2 for running this 
# code with a python interpreter.
# Importing libraries.
import matplotlib.image
from matplotlib import pyplot as plt
import random
import numpy as np
from numpy import mean
import cv2

# Global Variables
large_image = matplotlib.image.imread("Limage.jpg")
small_image = matplotlib.image.imread("Simage.jpg")
row, column = large_image.shape
smaller_x, smaller_y = small_image.shape

#  **Creating Lists and Dictionaries.**
row_list = []
column_list = []
Image_Points = {}
Max_Fitness_Values = {}
Mean_Fitness_Values = {}

# Class for Evolutionary Algorithum.
class Evolutionary_Algorithum:
    def __init__(self):
        self.population_size = 100                     # Defining Population size as 100.
        self.bigger = np.asarray(large_image)
        self.smaller = np.asarray(small_image)        # Convert image into array.

    def Initialization_Population(self):             # Randomly generated Population.
        for i in range(self.population_size):        # An Array of population points is generated through
            num1 = random.randint(0, row)            # zipping two lists.
            row_list.append(num1)
            num2 = random.randint(0, column)
            column_list.append(num2)

        Random_Population = list(zip(row_list, column_list))
        return Random_Population

    def Fitness_Sorting(self, Random_Population, Generation):

        Fitness_Values = {}
        CO_Relations = []

        for individual in Random_Population:

            template = [[0 for i in range(29)] for j in range(35)]

            m,n = individual
            for k in range(smaller_x):
                for l in range(smaller_y):
                    if m+k < 512 and n+l < 1024:
                        template[k][l] = large_image[m + k][n + l]

            correlation = self.correlation_coefficient(small_image, template)
            Fitness_Values[m, n] = correlation
            CO_Relations.append(correlation)

        for keys, values in Fitness_Values.items():     # Save individuals with maximum fitness points.
            if values == max(CO_Relations):
                Image_Points[keys] = max(CO_Relations)

        Max_Fitness_Values[Generation] = max(CO_Relations)  # Save maximum fitness points with respect to generation.
        Mean_Fitness_Values[Generation] = mean(CO_Relations)

        return Fitness_Values           # Returns a Dictionary with keys as Population and values as Corelation.

    def correlation_coefficient(self, small_image, Template):        # Helping Function
        numerator = np.mean((small_image - np.mean(small_image)) * (Template - np.mean(Template)))
        denominator = np.std(small_image) * np.std(Template)
        if denominator == 0:
            return 0
        else:
            result = numerator / denominator
            return result

    def Selection(self, Fitness_Values):              # Selecting fittest parent.
        Fitness_Values = sorted(Fitness_Values)
        return Fitness_Values

    # Crossover takes sorted population with individuals in decimal form.
    # Convert them to binary.
    # swap binary values from a specific point.
    # Send it for its mutation.
    def Crossover(self, Ranked_Population):
        Next_Generation = []

        if len(Ranked_Population) % 2 == 0:
           end = len(Ranked_Population)
        else:
            extra_point = (0,0)
            Ranked_Population.append(extra_point)
            end = len(Ranked_Population)

        for i in range(0, end, 2):
            # print(i)
            p1, p2 = Ranked_Population[i]
            q1, q2 = Ranked_Population[i+1]

            w = self.DecimalToBinary(p1, 0)
            x = self.DecimalToBinary(p2, 1)
            y = self.DecimalToBinary(q1, 0)
            z = self.DecimalToBinary(q2, 1)

            p = w
            q = y
            for i in x:    # Combining individual's point.
                p.append(i)
            for j in z:
                q.append(j)

            rnd_num = random.randint(0, 10)    # Random number to perform crossover
            for i in range(rnd_num, len(p)):
                p[i], q[i] = q[i], p[i]
            variable = p,q
            Next_Generation.append(variable)
        return Next_Generation

    # First mutation receives two binary number, then selects one random number and convert zero or 
    # one of both binary number with their opposites.
    # Adjust lengths of binary number and convert them to decimal number.
    # And return new population.
    def Mutation(self, Next_Generation):
        New_population_list = []
        for i in Next_Generation:
            p, q = i
            rnd_num = random.randint(0, 5)
            if p[rnd_num] == 0:
                p[rnd_num] = 1
            else:
                p[rnd_num] = 0
            if q[rnd_num] == 0:
                q[rnd_num] = 1
            else:
                q[rnd_num] = 0
            # divide
            count = 0
            count1 = 0
            a = []
            b = []
            c = []
            d = []

            for i in p:
                if count < 9:
                    a.append(i)
                    count += 1
                else:
                    b.append(i)

            for i in q:
                if count1 < 9:
                    c.append(i)
                    count1 += 1
                else:
                    d.append(i)

            a = self.Binary_to_decimal(a)
            b = self.Binary_to_decimal(b)
            c = self.Binary_to_decimal(c)
            d = self.Binary_to_decimal(d)

            num1 = (a,b)
            num2 = (c,d)
            New_population_list.append(num1)
            New_population_list.append(num2)
        return New_population_list

    def DecimalToBinary(self, num, check): # Helping function.
        binary = []
        while num >= 1:
            b = num % 2
            num = (num // 2)
            binary.append(b)

        if check == 0:
            while len(binary) != 9:
                binary.insert(0, 0)
        else:
            while len(binary) != 10:
                binary.insert(0, 0)
        # binary = "".join(num)
        return binary

    def Binary_to_decimal(self, p):      # Helping function.
        decimal = 0
        counter = 0
        for i in p[::-1]:
            decimal += 2 ** counter * int(i)
            counter += 1
        return decimal

    def End_Process(self):           # Check if Fitness value meet our requirement.
        Threshold = 0.95
        all_values = Max_Fitness_Values.values()
        if max(all_values) >= Threshold:

            return True
        else:
            return False

    def Image_Show(self):
        window_name = 'Image'
        Threshold = 0.8                    # Threshold for seeing rectangles on image.
        for individuals, fitness_value in Image_Points.items():
            if fitness_value >= Threshold:
                w = individuals[1]
                h = individuals[0]

                start_point = (w, h)
                end_point = (w + smaller_x, h + smaller_y)
                color = (255, 255, 0)
                thickness = 1
                image = cv2.rectangle(large_image, start_point, end_point, color, thickness)

                cv2.imshow(window_name, image)
        cv2.waitKey(0)


    def Start(self):
        Population = self.Initialization_Population()        # While population is random.
        Fitness_Values = self.Fitness_Sorting(Population, 0)
        Ranked_Population = self.Selection(Fitness_Values)
        Next_Generation = self.Crossover(Ranked_Population)
        Population = self.Mutation(Next_Generation)
        self.Image_Show()
        Generation_Limit = 0
        while Generation_Limit != 5000:                     # While population is newly generated.
            if not self.End_Process():
                Fitness_Values = self.Fitness_Sorting(Population,Generation_Limit)
                Ranked_Population = self.Selection(Fitness_Values)
                Next_Generation = self.Crossover(Ranked_Population)
                Population = self.Mutation(Next_Generation)
                Generation_Limit += 1
            else:
                break
        self.Image_Show()


    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^MAIN^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Evolutionary_Algorithum = Evolutionary_Algorithum()
Evolutionary_Algorithum.Start()
myplot = Max_Fitness_Values.items()
myplot1 = Mean_Fitness_Values.items()

x1, y1 = zip(*myplot)
x2, y2 = zip(*myplot1)
plt.plot(x1, y1, label = "Max_Fitness_Values")
plt.plot(x2, y2, label = "Mean_Fitness_Values")
plt.xlabel('Generation')
plt.ylabel('Fitness_Points')
plt.title('Fitness_Values with respect to Generations')
plt.show()
