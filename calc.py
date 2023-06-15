import sys

def add(args):
    total = 0
    for arg in args:
        try:
            value = float(arg)
            total += value
        except ValueError:
            print(f"Ignoring non-numeric value: {arg}")
    
    return total

def sub(args):
    if len(args) < 2:
        print("Please provide at least two values for subtraction.")
        return 0

    try:
        total = float(args[0])
        for arg in args[1:]:
            value = float(arg)
            total -= value
    except ValueError:
        print("Please provide numeric values for subtraction.")
        return 0
    
    return total

if __name__ == "__main__":
    operation = sys.argv[1]
    args = sys.argv[2:]

    if operation == "add":
        result = add(args)
    elif operation == "sub":
        result = sub(args)
    else:
        print("Invalid operation. Please choose 'add' or 'subtract'.")
        sys.exit(1)
    
    print(result)
