def check_list(x, elems):
    if x in elems or x == None:
        return True
    else:
        return False
    
def check_real(x):
    if isinstance(x, (float, int)) == True or x == None:
        return True
    return False
    
def check_natural(x):
    if (isinstance(x, int) == True and x >= 1) or x == None:
        return True
    return False

def check_natural_0(x):
    if (isinstance(x, int) == True and x >= 0) or x == None:
        return True
    return False

def cast_string(x, cast = 'float'):
    try:
        if cast == 'float':
            float(x)
            return float(x)
        elif cast == 'int':
            int(x)
            return int(x)
    except:
        return x
