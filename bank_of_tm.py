import requests
import pwn
import random

class web:
    pass

class register:
    def DB(self,name=0,passwd=0):
        names = dict()
        names[name]=passwd
        return names
    def reg(self):
        name = str(input('enter your name :'))
        passwd = str(input('enter passwd :'))
        re_type= str(input('re enter passwd :'))
        if re_type == passwd:
            passwd = passwd
            del re_type
            register.DB(self,name,passwd)
        else:
            print('passwd should match')
        main()
    def login(self,name,passwd):
        if passwd == register.DB(self)[name]:
            print(success)
print('welcome to bank of tm ')
def main():
    res = str(input('choose the options :\n1.register new user\n2.login\n3.exit\n ==>'))
    if res == '1':
        regobj.reg()
    elif res == '2':
        name = str(input('enter your name :'))
        passwd = str(input('enter your passwd'))
        regobj.login(name,passwd)
        print('currently in development')
    else:
        exit(0)
regobj = register()
main()
