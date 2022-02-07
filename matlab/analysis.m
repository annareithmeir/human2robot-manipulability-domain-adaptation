data_franka = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/panda/5000/manipulabilities.csv");
data_rhuman = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/rhuman/5000/manipulabilities.csv");
data_fanuc = csvread("/home/nnrthmr/CLionProjects/ma_thesis/data/mapping/toy_data/5000/manipulabilities.csv");
data_franka = reshape(data_franka,size(data_franka, 1),3,3);
data_rhuman = reshape(data_rhuman,size(data_rhuman,1),3,3);
data_fanuc = reshape(data_fanuc,size(data_fanuc,1),3,3);


eigs_franka=zeros(size(data_franka,1),3);
eigs_rhuman=zeros(size(data_rhuman,1),3);
eigs_fanuc=zeros(size(data_fanuc,1),3);

for i=1:size(data_franka,1)
    m1= reshape(data_franka(i,:),3,3);
    w = eig(m1);
    eigs_franka(i,:) = sort(w)';
    
    m2= reshape(data_rhuman(i,:),3,3);
    w2 = eig(m2);
    eigs_rhuman(i,:) = sort(w2)';
    
    m3= reshape(data_fanuc(i,:),3,3);
    w3 = eig(m3);
    eigs_fanuc(i,:) = sort(w3)';
end

figure(1)
subplot(2,3,1)
scatter3(eigs_franka(:,1),eigs_franka(:,2),eigs_franka(:,3),8,'filled','o')
title( 'Franka eigenvalues')
grid
alpha(.3)
subplot(2,3,2)
scatter3(eigs_rhuman(:,1),eigs_rhuman(:,2),eigs_rhuman(:,3),8,'filled','o')
title('RHuman eigenvalues')
grid
alpha(.3)
subplot(2,3,3)
scatter3(eigs_fanuc(:,1), eigs_fanuc(:,2), eigs_fanuc(:,3),8, 'filled','o')
title('Toy data eigenvalues')
grid
alpha(.3)


%cov1 = cov(eigs_franka');
tmp1 = eigs_franka' - mean(eigs_franka,1)';
cov1 = tmp1'*tmp1;
tmp2 = eigs_rhuman' - mean(eigs_rhuman,1)';
cov2 = tmp2'*tmp2;
tmp3 = eigs_fanuc' - mean(eigs_fanuc,1)';
cov3 = tmp3'*tmp3;

coveigs1=sort(eig(cov1));
coveigs1=coveigs1(4997:5000);
coveigs2=sort(eig(cov2));
coveigs2=coveigs2(4997:5000);
coveigs3=sort(eig(cov3));
coveigs3=coveigs3(4997:5000);


figure(1)
subplot(2,3,4)
plot(1:size(coveigs1,1), flip(coveigs1),'--o')
title('Franka cov spectrum ')
grid
subplot(2,3,5)
plot(1:size(coveigs2,1), flip(coveigs2),'--o')
title('RHuman cov spectrum ')
grid
subplot(2,3,6)
plot(1:size(coveigs3,1), flip(coveigs3),'--o')
title( 'Toy data cov spectrum ')
grid