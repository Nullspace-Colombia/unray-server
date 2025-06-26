
import os, platform

def print_title(): 
    """
        Print title 

        print a splash screen for 
    """
    clear_command = {
        "Darwin": "clear", 
        "Windows": "cls"
    }
    os.system(clear_command[platform.system()])
    
    print()
    print("                                         ")
    print("            _    _                       ")
    print("           | |  | |                      ")
    print("           | |  | |_ __  _ __ __ _ _   _ ")
    print("           | |  | | '_ \| '__/ _` | | | |")
    print("           | |__| | | | | | | (_| | |_| |")
    print("            \____/|_| |_|_|  \__,_|\__, |")
    print("                                    __/ |")
    print("                                   |___/ ")
    print("                                         ")
    print("** A RLLib framework for external simulation using Unreal Engine ")
    print("Created and Mantained by NullSpace")
    print(" ")
    print(" -- ")
    
