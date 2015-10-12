%% perfLogFile: 
function analyzeArthurPerf_x10(perfLogFile, p, e, epochNum)
    
    rawData=load(perfLogFile);
    reshapedData = reshape(rawData ./ epochNum, e,p)';
    FigHandle = figure('Position', [100, 100, 900, 595]);
    xlhand = get(gca,'xlabel')
    set(xlhand,'string','X','fontsize',20)
    if(p > 1)
        bar(1:p, reshapedData, 0.5, 'stack');
    elseif(p == 1)
        bar(1:2, repmat(reshapedData,2,1), 0.5, 'stack'); %% hack to get a single column stacked bar plot
    else
        print "wrong p"
    end
    subMat = zeros(size(reshapedData));
    subMat(:,8) = reshapedData(:,12); % get the 12th column (i.e., applyUpdate timer) 
    reshapedData = reshapedData - subMat;
     trainTime = sum(reshapedData(:,6)) % every row is per place stat, every column is an event
     maxTime = max(sum(reshapedData,2)) % sum along the 2nd dimension(i..e, the column dimention) and get max data as the time spent
    legend('load', 'bcast', 'deser\_weights', 'sel\_train\_data', 'pull\_w', 'train','s\_updates', 'push\_updates', 'report\_train\_err','test','report\_test\_err', 'apply\_update');
    title('\fontsize{16}Where did time go ?')
    xlabel('\fontsize{16}Number of learners')
    ylabel('\fontsize{16}Time spent in training one epoch')
end
