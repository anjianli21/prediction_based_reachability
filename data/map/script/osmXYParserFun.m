%%%%%%%% usage: Lines = osmXYParserFun(filename, IS_SHOW) or 
%%%%%%%% [CurbPts, LanePts, StopPts] = osmXYParserFun(filename, IS_SHOW). 
%%%%%%%% The input file xxx.osm_xy can be downloaded from GitLab.
function varargout = osmXYParserFun(filename, IS_SHOW)
if nargin == 0
    OSMRoot = '/home/mscsim/Gabor/drone_dataset_maps/maps'; 
    sCapture = 'USA_Roundabout_FT'; 
    DataFolder = fullfile(OSMRoot, sprintf( 'DR_%s', sCapture ) ); 
    TmpList = dir( fullfile(DataFolder, '*.osm_xy') ); 
    filename = fullfile(TmpList(1).folder, TmpList(1).name); 
    IS_SHOW = 1;
end

Lines = [];
DOMnode = parseXML(filename);
data = DOMnode.Children;
AllXY = [];
Lines0 = [];
for id = 1 : 1 : length(data)
    tmpInfo = data(id);
    if strcmpi(tmpInfo.Name, 'node')
        Attributes = tmpInfo.Attributes;
        llh = zeros(3, 1);
        for i = 1 : 1 : length(Attributes)
            tmp = Attributes(i);
            if strcmpi(tmp.Name, 'id')
                llh(3) = str2num(tmp.Value);
            end
            if strcmpi(tmp.Name, 'x')
                llh(1) = str2num(tmp.Value);
            end
            if strcmpi(tmp.Name, 'y')
                llh(2) = str2num(tmp.Value);
            end
            bTest = 1;
        end
        if ~isempty(llh)
            AllXY(:, end+1) = llh;
        end
        bTest = 1;
    end
    
    if strcmpi(tmpInfo.Name, 'way')
        tmp = tmpInfo.Children;
        Idx = [];
        sType = [];
        for id = 1 : 1 : length(tmp)
            if strcmpi(tmp(id).Name, 'tag')
                if strcmpi( tmp(id).Attributes(1).Value, 'type' )
                    sType = tmp(id).Attributes(2).Value;
                    bTest = 1;
                end
            end
            if strcmpi(tmp(id).Name, 'nd')
                Idx(end+1) = str2num( tmp(id).Attributes.Value );
                bTest = 1;
            end
        end
        if ~isempty(Idx)
            tmp = [];
            tmp.idx = Idx;
            tmp.type = sType;
            Lines0 = [Lines0 tmp];
            bTest = 1;
        end
        bTest = 1;
        
    end
end
for id = 1 : 1 : length(Lines0)
    tmpInfo = Lines0(id);
    sType = tmpInfo.type;
    Idx = tmpInfo.idx;
    [NNIdx, ~] = knnsearch(AllXY(3, :)', Idx');
    xy = AllXY(1:2, NNIdx);
    tmpInfo.xy = xy;
    tmpInfo = rmfield(tmpInfo, 'idx');
    if strcmpi(sType, 'curbstone') | strcmpi(sType, 'guard_rail') | strcmp(sType, 'rail')
        sType = 'curb'; 
    end
    if strcmpi(sType, 'line_thin') | strcmpi(sType, 'line_thick')
        sType = 'lane'; 
    end
    if strcmpi(sType, 'stop_line') | strcmpi(sType, 'pedestrian_marking')
        sType = 'stop'; 
    end
    tmpInfo.type = sType; 
    if strcmpi(sType, 'curb') | strcmpi(sType, 'lane') | strcmpi(sType, 'stop')
        Lines = [Lines tmpInfo];
    end
end


if nargout == 1 | nargout == 0
    varargout{1} = Lines;
    if IS_SHOW
        figure;
        hold on;
        grid on;
        axis equal;
        box on;
        xlabel('X/m');
        ylabel('Y/m');
        for id = 1 : 1 : length(Lines)
            tmpInfo = Lines(id);
            xy = tmpInfo.xy;
            if strcmpi(tmpInfo.type, 'curb')
                plot(xy(1, :), xy(2, :), 'k.-');
            end
            if strcmpi(tmpInfo.type, 'lane')
                plot(xy(1, :), xy(2, :), 'b.-');
            end
            if strcmpi(tmpInfo.type, 'stop')
                plot(xy(1, :), xy(2, :), 'r.-');
            end
            bTest = 1;
        end
    end
end
if nargout == 3
    CurbPts = []; 
    LanePts = []; 
    StopPts = []; 
    res = 0.4; 
    for id = 1 : 1 : length(Lines)
        tmpInfo = Lines(id);
        xy = tmpInfo.xy;
        xy = DenseInterpFun(xy, res); 
        if strcmpi(tmpInfo.type, 'curb')
            CurbPts = [CurbPts xy]; 
        end
        if strcmpi(tmpInfo.type, 'lane')
            LanePts = [LanePts xy]; 
        end
        if strcmpi(tmpInfo.type, 'stop')
            StopPts = [StopPts xy]; 
        end
        bTest = 1;
    end
    varargout{1} = CurbPts; 
    varargout{2} = LanePts; 
    varargout{3} = StopPts; 
    if IS_SHOW
        figure;
        hold on;
%         grid on;
        axis equal;
%         box on;
%         xlabel('X/m');
%         ylabel('Y/m');
        if ~isempty(CurbPts)
            plot(CurbPts(1, :), CurbPts(2, :), 'k.');
        end
        if ~isempty(LanePts)
            plot(LanePts(1, :), LanePts(2, :), 'b.');
        end
        if ~isempty(StopPts)
            plot(StopPts(1, :), StopPts(2, :), 'r.');
        end
    end
end

function [ dataNew ] = DenseInterpFun( data, res )
if size(data, 1) ~= 2
    error('DenseInterpFun error!'); 
end
dataNew = []; 
for id = 1 : 1 : length(data)-1
    pt0 = data(:, id); 
    pt1 = data(:, id+1); 
    Dist = norm(pt1 - pt0); 
    if Dist < 1e-8
        continue; 
    end
    nLen = ceil(Dist/res); 
    ratio = (0:1:nLen)/nLen; 
    x = (1-ratio)*pt0(1) + ratio*pt1(1); 
    y = (1-ratio)*pt0(2) + ratio*pt1(2);
    % plot(x, y, 'r.'); 
    tmp = [x; y]; 
    dataNew = [dataNew tmp]; 
end
dataNew = unique(dataNew', 'rows', 'stable')'; 

function theStruct = parseXML(filename)
% PARSEXML Convert XML file to a MATLAB structure.
try
   tree = xmlread(filename);
catch
   error('Failed to read XML file %s.',filename);
end

% Recurse over child nodes. This could run into problems 
% with very deeply nested trees.
try
   theStruct = parseChildNodes(tree);
catch
   error('Unable to parse XML file %s.',filename);
end


% ----- Local function PARSECHILDNODES -----
function children = parseChildNodes(theNode)
% Recurse over node children.
children = [];
if theNode.hasChildNodes
   childNodes = theNode.getChildNodes;
   numChildNodes = childNodes.getLength;
   allocCell = cell(1, numChildNodes);

   children = struct(             ...
      'Name', allocCell, 'Attributes', allocCell,    ...
      'Data', allocCell, 'Children', allocCell);

    for count = 1:numChildNodes
        theChild = childNodes.item(count-1);
        children(count) = makeStructFromNode(theChild);
    end
end

% ----- Local function MAKESTRUCTFROMNODE -----
function nodeStruct = makeStructFromNode(theNode)
% Create structure of node info.

nodeStruct = struct(                        ...
   'Name', char(theNode.getNodeName),       ...
   'Attributes', parseAttributes(theNode),  ...
   'Data', '',                              ...
   'Children', parseChildNodes(theNode));

if any(strcmp(methods(theNode), 'getData'))
   nodeStruct.Data = char(theNode.getData); 
else
   nodeStruct.Data = '';
end

% ----- Local function PARSEATTRIBUTES -----
function attributes = parseAttributes(theNode)
% Create attributes structure.

attributes = [];
if theNode.hasAttributes
   theAttributes = theNode.getAttributes;
   numAttributes = theAttributes.getLength;
   allocCell = cell(1, numAttributes);
   attributes = struct('Name', allocCell, 'Value', ...
                       allocCell);

   for count = 1:numAttributes
      attrib = theAttributes.item(count-1);
      attributes(count).Name = char(attrib.getName);
      attributes(count).Value = char(attrib.getValue);
   end
end
