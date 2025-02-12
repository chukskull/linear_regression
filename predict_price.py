import json
import sys


def get_thetas():
    try:
        with open('thetas.json', 'r') as read_file:
            data = json.load(read_file)
    except FileNotFoundError:
        return 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(-1)
    return data['theta0'], data['theta1']



def price_estimation():
    theta0, theta1 = get_thetas()
    
    while True:
        user_input = input("Enter car mileage: ")
        try:
            mileage = float(user_input)
            if mileage < 0:
                print("Mileage cannot be negative. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value for mileage.")
    estimated_price = theta0 + theta1 * mileage
    print(f"Estimated Price: {estimated_price:.2f}")



if __name__ == '__main__':
    price_estimation()