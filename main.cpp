#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
using namespace std;
using namespace Eigen;

MatrixXd relu(const MatrixXd& x) {
    return x.cwiseMax(0);
}

MatrixXd relu_deriv(const MatrixXd& x) {
    return (x.array() > 0).cast<double>();
}

MatrixXd softmax(const MatrixXd& x) {
    MatrixXd y(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); i++) {
        double mx = x.row(i).maxCoeff();
        VectorXd exps = (x.row(i).array() - mx).exp();
        y.row(i) = exps / exps.sum();
    }
    return y;
}

double crossEntropyLoss(const MatrixXd& pred, const MatrixXd& target) {
    double loss = 0.0;
    int m = pred.rows();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < pred.cols(); j++) {
            loss -= target(i, j) * log(pred(i, j) + 1e-8);
        }
    }
    return loss / m;
}

MatrixXd crossEntropyGrad(const MatrixXd& pred, const MatrixXd& target) {
    return (pred - target) / pred.rows();
}

class Dense {
public:
    MatrixXd w, b, x, out;
    Dense(int in, int outSize) {
        w = MatrixXd::Random(in, outSize) * sqrt(2.0 / in);
        b = MatrixXd::Zero(1, outSize);
    }
    MatrixXd forward(const MatrixXd& in) {
        x = in;
        out = (x * w).rowwise() + b;
        return out;
    }
    MatrixXd backward(const MatrixXd& grad, double lr) {
        MatrixXd gradW = x.transpose() * grad;
        MatrixXd gradB = grad.colwise().sum();
        MatrixXd gradX = grad * w.transpose();
        w -= lr * gradW;
        b -= lr * gradB;
        return gradX;
    }
};

class NeuralNet {
public:
    Dense l1, l2;
    MatrixXd z1, a1, z2, a2;
    NeuralNet() : l1(2, 10), l2(10, 2) {}
    MatrixXd forward(const MatrixXd& X) {
        z1 = l1.forward(X);
        a1 = relu(z1);
        z2 = l2.forward(a1);
        a2 = softmax(z2);
        return a2;
    }
    void backward(const MatrixXd& y, double lr) {
        MatrixXd grad = crossEntropyGrad(a2, y);
        MatrixXd grad2 = l2.backward(grad, lr);
        MatrixXd grad1 = grad2.cwiseProduct(relu_deriv(z1));
        l1.backward(grad1, lr);
    }
    double accuracy(const MatrixXd& pred, const MatrixXd& y) {
        int correct = 0;
        for (int i = 0; i < pred.rows(); i++) {
            int pi = 0, yi = 0;
            pred.row(i).maxCoeff(&pi);
            y.row(i).maxCoeff(&yi);
            if (pi == yi) correct++;
        }
        return static_cast<double>(correct) / pred.rows();
    }
};

int main(){
    int n = 200;
    MatrixXd X(n, 2), Y(n, 2);
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0, 1);
    for (int i = 0; i < n / 2; i++){
        X(i, 0) = dist(gen) + 1.0;
        X(i, 1) = dist(gen) + 1.0;
        Y(i, 0) = 1; Y(i, 1) = 0;
    }
    for (int i = n / 2; i < n; i++){
        X(i, 0) = dist(gen) - 1.0;
        X(i, 1) = dist(gen) - 1.0;
        Y(i, 0) = 0; Y(i, 1) = 1;
    }
    shuffle(idx.begin(), idx.end(), gen);
    MatrixXd Xs(n, 2), Ys(n, 2);
    for (int i = 0; i < n; i++){
        Xs.row(i) = X.row(idx[i]);
        Ys.row(i) = Y.row(idx[i]);
    }
    int trainSize = static_cast<int>(0.8 * n);
    MatrixXd XTrain = Xs.topRows(trainSize);
    MatrixXd YTrain = Ys.topRows(trainSize);
    MatrixXd XTest = Xs.bottomRows(n - trainSize);
    MatrixXd YTest = Ys.bottomRows(n - trainSize);
    NeuralNet nn;
    for (int epoch = 0; epoch < 1000; epoch++){
        MatrixXd pred = nn.forward(XTrain);
        double loss = crossEntropyLoss(pred, YTrain);
        nn.backward(YTrain, 0.01);
        if (epoch % 100 == 0)
            cout << "Epoch " << epoch << " loss: " << loss << " acc: " << nn.accuracy(pred, YTrain) << "\n";
    }
    MatrixXd testPred = nn.forward(XTest);
    cout << "Test acc: " << nn.accuracy(testPred, YTest) << "\n";
    return 0;
}
