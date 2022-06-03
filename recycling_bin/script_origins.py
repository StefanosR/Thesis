import sys
import os 

def myscript(iteration_number):
    xfile_name = "x%d.csv" % iteration_number
    with open(xfile_name, "w") as xf:
        print("Iteration:%d" % iteration_number)
           
def main(unused_command_line_args):
    for i in range(10):
        myscript(i)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
    
    