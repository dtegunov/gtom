function interleaved = complexinterleaved(z)

[n1, n2, n3] = size(z);
z_real = real(z);
z_imag = imag(z);
if (n3 == 1)
    interleaved = zeros(n1*2, n2);
else
    interleaved = zeros(n1*2, n2, n3);
end;
newrow = 1;

for row = 1:n1
    interleaved(newrow,:,:) = z_real(row,:,:);
    interleaved(newrow + 1,:,:) = z_imag(row,:,:);
    newrow = newrow + 2;
end

end