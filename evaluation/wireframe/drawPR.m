%clear;
figure(2);

load('0.2_0.5.mat')
p = sumprecisions ./ nsamples;
r = sumrecalls ./ nsamples;
disp([p r]);
plot(r, p, '-^', 'LineWidth', 2)
hold on

% load('output_lsd_ours.mat')
% p = sumprecisions ./ nsamples;
% r = sumrecalls ./ nsamples;
% plot(r(1:19), p(1:19), '-^', 'LineWidth', 2)
% hold on
% 
% load('output_mcmlsd.mat')
% p = sumprecisions ./ nsamples;
% r = sumrecalls ./ nsamples;
% plot(r, p, '-d', 'LineWidth', 2)
% 
% load('output_3.mat')
% p = sumprecisions ./ nsamples;
% r = sumrecalls ./ nsamples;
% plot(r, p, '-*', 'LineWidth', 2)
% 
% load('output_hough.mat')
% p = sumprecisions ./ nsamples;
% r = sumrecalls ./ nsamples;
% plot(r(2:10), p(2:10), '-v', 'LineWidth', 2)

xlim([0 1])
ylim([0 1])
set(gca, 'xtick', [0:0.1:1])
set(gca, 'ytick', [0:0.1:1])
set(gca, 'fontsize', 20)
grid on

xlabel('Recall','fontsize',20);
ylabel('Precision','fontsize',20);
% legend('LSD','MCMLSD','Our Method','Heatmap+Hough');
legend('Our Method');
hold off;
