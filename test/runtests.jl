using FastConv
using Test

A = zeros(5,5)
A[3,3] = 1
k = rand(3,3)
Ac = convn(A,k)
@test Ac[3:5,3:5] == k && all(a->a==0, Ac[:,[1,2,6,7]]) && all(a->a==0, Ac[[1,2,6,7], :])

A = zeros(5,6,7)
A[3,4,5] = 1
k = rand(3,3,3)
Ac = convn(A,k)
@test Ac[3:5,4:6,5:7] == k &&
    all(a->a==0, Ac[[1,2,6,7], :, :]) &&
    all(a->a==0, Ac[:, [1,2,3,7,8], :]) &&
    all(a->a==0, Ac[:, :, [1,2,3,4,8,9]])

A = randn(5,5)
k = ones(1,1)
Ac = convn(A,k)
@test Ac == A

A = randn(5,5)
k = zeros(3,3)
Ac = convn(A,k)
@test Ac == zeros(7,7)

A = randn(5,5)
k1 = rand(3,3)
k2 = rand(3,3)
A1_2 = convn(convn(A,k1), k2)
A_12 = fastconv(A, fastconv(k1, k2))
@test A1_2 \approx A_12

# Test Array Views
u = Float32.(reshape(1:12,(3,4)))
v = Float32.(reshape(1:6,(3,2)))
y_loop = [convn(u[i,:],v[i,:]) for i in 1:size(u)[1]]
y_broadcast = convn.(eachrow(u),eachrow(v))
@test y_loop ≈ y_broadcast
