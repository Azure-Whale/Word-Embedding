import string
import math

# test=2
# print(math.log(4,2))
a = input()
b = input()
Result = -(a * math.log(a, 2) + b * math.log(b,2))
print(Result)
x = input()


def Ex(string):
    if string:
        print(string)
        # print('Nothing will be printed!')
    else:
        print('Nothing will be printed!')


# Slices Operation of String
string = 'hello world'
Ex(string[-1:0:-2])  # Back from the end of the string, the interval is 2
Ex(string[-1:0:+ 2])  # No putput
Ex(string[-1::-1])  # Reversed
Ex(string[-1:-1])  # No output
Ex(string[1:1])  # No output

# Common Function for string
Her = 'ZhangZiXing'
print(Her.index('ngZ'))  # If the string you are going to find exists,
# then it will return the index of the first char in the string
print(Her.count('Z'))  # Return the number of the char or the string in origin string
print(Her.count('Z', 2, 4))  # Return the number of the char or the string in origin string
print(Her.find('Z'))  # Return the index of the char first found in the origin string
print(Her.split('Z'))  # The string will be split and stored in a list
# Watch out, if the first element is split,  there would be '' left
'''Here, Note that the sep means separator'''
print(Her.split(sep='Z'))  # all the string will be split into pieces according to sep, not for the first element only
print(Her.strip())  # When there is no parameters, the default value is ' '.
print('azurewhale1127', 'gmail', sep='@')
print(Her.replace('Zhang', 'Xiang').replace('ZiXing', 'KeZheng'))  # Here, there are two summons of
# replace() in succession, think about it.

'''
count()
capitalize()
startwith
endswith
isalpha
isdigit
upper
isupper
islower
stripe
find
lstripe
rstripe
'''
