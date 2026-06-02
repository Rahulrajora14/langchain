from typing import TypedDict

class person(TypedDict) :
    name : str 
    rollno : int

new_person : person = { 'name' : 'rahul' , 'rollno': '35' }

print(new_person)
