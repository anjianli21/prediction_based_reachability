% clear all;
% load('./segment_results_DR_USA_Roundabout_SR/Roundabout_DR_USA_Roundabout_SR.mat');
% load('./segment_results_DR_USA_Intersection_EP0/New_reference_EP0.mat');
load('./New_reference_EP0.mat');


%%
% [maps.CurbPts, maps.LanePts, maps.StopPts] = osmXYParserFun('./DR_USA_Intersection_EP0.osm_xy', 1);
[maps.CurbPts, maps.LanePts, maps.StopPts] = osmXYParserFun('./DR_USA_Roundabout_FT.osm_xy', 1);


%%
% if there is a roundabout structure in the map
scatter(roundabout.center(1), roundabout.center(2));
hold on;
plot(roundabout.fitted_circle_curves.para_path(:,1), roundabout.fitted_circle_curves.para_path(:,2), 'g');
scatter(roundabout.center(1) + roundabout.reference_circle_radius*cos(roundabout.merge_angle_list/180*pi), ...
        roundabout.center(2) + roundabout.reference_circle_radius*sin(roundabout.merge_angle_list/180*pi), 'b');
scatter(roundabout.center(1) + roundabout.reference_circle_radius*cos(roundabout.demerge_angle_list/180*pi), ...
        roundabout.center(2) + roundabout.reference_circle_radius*sin(roundabout.demerge_angle_list/180*pi), 'm');

%%
figure, hold on;
for i=1:length(Segmented_reference_path)
    if Segmented_reference_path(i).branchID(2) == -1
        plot(Segmented_reference_path(i).para_path(:,1), Segmented_reference_path(i).para_path(:,2), 'b');
        scatter(Segmented_reference_path(i).para_path(end,1), Segmented_reference_path(i).para_path(end,2), 'b+');
    elseif Segmented_reference_path(i).branchID(1) == -1
        plot(Segmented_reference_path(i).para_path(:,1), Segmented_reference_path(i).para_path(:,2), 'm');
        scatter(Segmented_reference_path(i).para_path(end,1), Segmented_reference_path(i).para_path(end,2), 'm+');
    else
        plot(Segmented_reference_path(i).para_path(:,1), Segmented_reference_path(i).para_path(:,2), 'c');
        scatter(Segmented_reference_path(i).para_path(1,1), Segmented_reference_path(i).para_path(1,2), 'co');
        scatter(Segmented_reference_path(i).para_path(end,1), Segmented_reference_path(i).para_path(end,2), 'c+');
    end
end


