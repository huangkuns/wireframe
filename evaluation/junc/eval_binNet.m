exp_name = '1'
epoch = '16'
root_dir = '../../result/junc/'
root_folder = [root_dir exp_name '/' epoch];
label_folder = 'junc_label';

Dist= 0.01;

pr_iter = [];
rc_iter = [];

for thresh = 0:9
    
    exp_folder = [root_folder '/' num2str(thresh)];
    filenames = dir([exp_folder '/' '*.mat']);
    num_of_image = length(filenames);
    
    prec = 0;
    recall = 0;
    
    for i = 1:num_of_image
        resmatfile = filenames(i).name;
        %disp(resmatfile);
        
        res = load([exp_folder '/' resmatfile]);
        
        h= res.h;
        w = res.w;
        junctions = res.junctions;
        thetas = res.thetas;
        if ~iscell(thetas)
            thetas = mat2cell(thetas,ones(1,size(thetas,1)));
        end
        
        labelfile = resmatfile;
        data = load([label_folder '/' labelfile]);
        gt = data.label;        
        
        g = zeros([h, w]);
        p = zeros([h, w]);
        for t0 = 1:size(junctions, 1)
            theta = thetas{t0};
            if length(theta) >= 2 && length(theta) <= 5 % select junctions
                tx = junctions(t0, 1); ty = junctions(t0, 2);
                tx = int16(tx); ty = int16(ty);
                if tx <= 0
                    tx = 1;
                end
                if tx > w
                    tx = w;
                end
                if ty <= 0
                    ty = 1;
                end
                if ty > h
                    ty = h;
                end
                p(ty, tx) = 1;
            end
        end
        
        for t1 = 1: size(gt, 1)
            tx = gt(t1, 1); ty = gt(t1, 2);
            tx = int16(tx); ty = int16(ty);
            if tx <= 0
                tx = 1;
            end
            if tx > w
                tx = w;
            end
            if ty <= 0
                ty = 1;
            end
            if ty > h
                ty = h;
            end
            g(ty, tx) = 1;
        end
        
        [matchE1,matchG1] = correspondPixels(p,g,Dist);
        matchE = double(matchE1>0);matchG = double(matchG1>0);
        cntR = sum(matchG(:)); sumR = sum(g(:));
        cntP = nnz(matchE); sumP = nnz(p);
        recall = recall + cntR/(double(sumR) + 1e-20);
        prec = prec + cntP/(double(sumP)  + 1e-20);
        %disp([cntR/double(sumR), cntP/double(sumP)]);
    end
    prec = prec/num_of_image;
    recall = recall/num_of_image;
    disp([prec, recall]);
    
    pr_iter = [pr_iter, prec];
    rc_iter = [rc_iter, recall];
end
data = [pr_iter; rc_iter];
disp(data);

figure(1);
plot(rc_iter, pr_iter, 'bd-','LineWidth',2);
xlim([0,1]);
ylim([0,1]);
set(gca,'XTick',0:0.1:1,'YTick',0:0.1:1,'fontsize',16);
grid on;
xlabel('Recall','fontsize',16);
ylabel('Precision','fontsize',16);

save(['plots/' exp_name '_' epoch '.mat'],'pr_iter','rc_iter');
