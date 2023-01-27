class bcolors:
    """TODO make this universal"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printc(text, color):
    print(color + text + bcolors.ENDC)

def printc_warn(text):
    printc(text, bcolors.WARNING)

def printc_fail(text):
    printc(text, bcolors.FAIL)

