import environment
import numpy as np
import pybullet as p

#Testing if pybullet handles orientations the way I suspect

# https://github.com/bulletphysics/bullet3/blob/master/src/Bullet3Common/b3Quaternion.h
# Line 307
def multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return [x,y,z,w]

def inverse(q):
    x,y,z,w = q
    return [-x, -y, -z, w]


# To rotate a 3D vector by a unit quaternion you perform v' = q*v*q^-1
# But to compose rotations (ie do one then the other ie add rotations) you just do q = q2*q1 (q1 then q2)
# v' = q1*v*q1^-1,  v'' = q2*v'*q2^-1   =>   v'' = q2*q1*v*q1^-1*q2^-1   =>   (q2*q1)*v*(q2*q1)^-1 
# Pybullet sotres orientation as a quaternion NOT a vector in 3D space. Therefore I believe that
# orientation is stored as the rotation relative to the initial standard reference frame. If this
# assumption is true the following tests should all pass

env = environment.FrankaArmEnvironment()
print("")

all_pass = True

# Test 1 - rotating from the identity results in just the rotation orientation
start_orn = [0,0,0,1] #identity quaternion
end_orn = env.getQuaternionFromEuler([0,0,np.pi/2])

actual = multiply(start_orn, end_orn)
desired = end_orn

correct = np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 1: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")


# Test 2 - rotating by the identity has no effect
start_orn = env.getQuaternionFromEuler([0,0,np.pi/3])
rotate_orn = [0,0,0,1] #identity quaternion

actual = multiply(rotate_orn, start_orn)
desired = start_orn

correct = np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 2: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")


# Test 3 - rotating quaternion p by a quaternion q results in orientation qp
start_orn = env.getQuaternionFromEuler([0, np.pi/4, np.pi/4])
rotate_orn = env.getQuaternionFromEuler([0, 0, np.pi/2])
end_orn = env.getQuaternionFromEuler([0.0, np.pi/4, 3*np.pi/4])

actual = multiply(rotate_orn, start_orn)
desired = end_orn

correct = np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 3: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")


# Test 4 - rotating quaternion p by a quaternion q does not results in orientation pq, quaternion multiplication is not commutative
start_orn = env.getQuaternionFromEuler([0, np.pi/4, np.pi/4])
rotate_orn = env.getQuaternionFromEuler([0, 0, np.pi/2])
end_orn = env.getQuaternionFromEuler([0.0, np.pi/4, 3*np.pi/4])

actual = multiply(start_orn, rotate_orn)
desired = end_orn

correct = not np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 4: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")


# Test 5 - rotating vector v by a quaternion q does results in vector qvq^-1
vec = [1,2,3]
pure_quat = [1,2,3,0] #convert v to a pure quaternion

rotate = env.getQuaternionFromEuler([0, np.pi, 0])

actual = multiply(multiply(rotate, pure_quat), inverse(rotate))
actual = actual[:3] #to get the rotated vector we drop the w component which we added by making it a pure quaternion
desired = [-1, 2, -3]

correct = np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 5: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")


# Test 6 - rotating vector v by a quaternion q does results in vector qvq^-1
vec = [1,2,3]
pure_quat = [1,2,3,0] #convert v to a pure quaternion

rotate = env.getQuaternionFromEuler([0, 0, np.pi/2])

actual = multiply(multiply(rotate, pure_quat), inverse(rotate))
actual = actual[:3] #to get the rotated vector we drop the w component which we added by making it a pure quaternion
desired = [-2, 1, 3]

correct = np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 6: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")



# Test 7 - rotating quaternion p by a quaternion q does not results in quaternion qvq^-1
start_orn = env.getQuaternionFromEuler([0, np.pi/4, np.pi/4])
rotate_orn = env.getQuaternionFromEuler([0, 0, np.pi/2])
end_orn = env.getQuaternionFromEuler([0.0, np.pi/4, 3*np.pi/4])

actual = multiply(multiply(rotate_orn, start_orn), inverse(rotate_orn))
desired = end_orn

correct = not np.allclose(actual, desired)
print(actual)
print(desired)
print("Test 6: " + "Pass" if correct else "FAIL")
all_pass = all_pass and correct
print("")


print("ALL TESTS PASSED" if all_pass else "NOT ALL TESTS PASSED")
print("")