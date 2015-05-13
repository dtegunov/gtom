function [ PC, signals ] = hacked_pca( data )

data = data';
[M, N] = size(data);

mn = mean(data, 2);
data = data - repmat(mn, 1, N);

Y = data' / sqrt(N - 1);

[u, S, PC] = svd(Y);

signals = PC' * data;

end

