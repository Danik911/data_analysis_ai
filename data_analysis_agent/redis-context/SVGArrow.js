import React, { useMemo } from 'react';
import './FlowArrow.css';

const SVGArrow = ({ flow, type, nodes, onClick }) => {
  // Find the source and target nodes
  const findNodePosition = (id) => {
    const node = nodes.find(n => n?.id === id);
    if (!node) return null;
    
    // Get node dimensions based on type
    let width = 160; // Default width
    let height = 80;  // Default height
    
    if (id.includes('supplier')) {
      width = 160;
      height = 120;
    } else if (id.includes('inventory')) {
      width = 150;
      height = 100;
    } else if (id.includes('customer')) {
      width = 180;
      height = 120;
    } else if (id.includes('process')) {
      width = 170;
      height = 100;
    }
    
    return { 
      id,
      x: node.position.x, 
      y: node.position.y,
      width,
      height
    };
  };

  const fromNode = findNodePosition(flow.from);
  const toNode = findNodePosition(flow.to);

  // Early return with null if nodes not found - move this after all hooks
  // to avoid conditional hook calls

  // Calculate potential connection ports along each side of the nodes
  const getNodePorts = (node) => {
    if (!node) return [];
    
    const positions = [];
    
    // Top side - 3 ports
    for (let i = 1; i <= 3; i++) {
      positions.push({
        side: 'top',
        x: node.x + (node.width * i / 4),
        y: node.y
      });
    }
    
    // Bottom side - 3 ports
    for (let i = 1; i <= 3; i++) {
      positions.push({
        side: 'bottom',
        x: node.x + (node.width * i / 4),
        y: node.y + node.height
      });
    }
    
    // Left side - 2 ports
    for (let i = 1; i <= 2; i++) {
      positions.push({
        side: 'left',
        x: node.x,
        y: node.y + (node.height * i / 3)
      });
    }
    
    // Right side - 2 ports
    for (let i = 1; i <= 2; i++) {
      positions.push({
        side: 'right',
        x: node.x + node.width,
        y: node.y + (node.height * i / 3)
      });
    }
    
    return positions;
  };

  // Find the best connection ports
  const findBestConnectionPorts = (fromNodeArg, toNodeArg) => {
    if (!fromNodeArg || !toNodeArg) {
      return { 
        fromPort: null, 
        toPort: null 
      };
    }
    
    const fromPorts = getNodePorts(fromNodeArg);
    const toPorts = getNodePorts(toNodeArg);
    
    if (fromPorts.length === 0 || toPorts.length === 0) {
      return { 
        fromPort: null, 
        toPort: null 
      };
    }
    
    // Calculate the combination with minimum distance
    let bestDistance = Infinity;
    let bestFromPort = null;
    let bestToPort = null;
    
    // Prioritize horizontal connections for better visual flow
    for (const fromPort of fromPorts) {
      for (const toPort of toPorts) {
        // Avoid connecting to the same side (e.g., right to right)
        if (fromPort.side === toPort.side) {
          continue;
        }
        
        // Calculate distance between ports
        const dx = toPort.x - fromPort.x;
        const dy = toPort.y - fromPort.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Apply weights based on sides for better flow direction
        let weight = 1.0;
        
        // Prefer right-to-left connections (natural flow)
        if (fromPort.side === 'right' && toPort.side === 'left') {
          weight = 0.7; // Prefer this combination
        }
        
        // Avoid top-to-bottom if possible (unless vertical layout is needed)
        if ((fromPort.side === 'bottom' && toPort.side === 'top') ||
            (fromPort.side === 'top' && toPort.side === 'bottom')) {
          weight = 1.2;
        }
        
        const weightedDistance = distance * weight;
        
        if (weightedDistance < bestDistance) {
          bestDistance = weightedDistance;
          bestFromPort = fromPort;
          bestToPort = toPort;
        }
      }
    }
    
    // If specific node types, override with better connections
    if (fromNodeArg.id?.includes('inventory') && toNodeArg.id?.includes('process')) {
      const fromRightPorts = fromPorts.filter(p => p.side === 'right');
      const toLeftPorts = toPorts.filter(p => p.side === 'left');
      
      if (fromRightPorts.length > 0 && toLeftPorts.length > 0) {
        bestFromPort = fromRightPorts[0];
        bestToPort = toLeftPorts[0];
      }
    }
    
    return { fromPort: bestFromPort, toPort: bestToPort };
  };

  // Use useMemo unconditionally
  const portInfo = useMemo(() => {
    return findBestConnectionPorts(fromNode, toNode);
  }, [fromNode, toNode]);
  
  const { fromPort, toPort } = portInfo;

  // Create SVG path for curved arrow - ensure this is called unconditionally
  const createPath = (fromPortArg, toPortArg) => {
    if (!fromPortArg || !toPortArg) {
      // Default fallback values for path if ports are not available
      return {
        path: "",
        midPoint: { x: 0, y: 0 }
      };
    }
    
    // Extract coordinates
    const x1 = fromPortArg.x;
    const y1 = fromPortArg.y;
    const x2 = toPortArg.x;
    const y2 = toPortArg.y;
    
    // Calculate control points for Bezier curve
    const dx = x2 - x1;
    const dy = y2 - y1;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Control point offset - increase for more curve
    const curveOffset = distance * 0.4;
    
    let cx1, cy1, cx2, cy2;
    
    // Adjust control points based on the sides the ports are on
    if (fromPortArg.side === 'right') {
      cx1 = x1 + curveOffset;
      cy1 = y1;
    } else if (fromPortArg.side === 'left') {
      cx1 = x1 - curveOffset;
      cy1 = y1;
    } else if (fromPortArg.side === 'top') {
      cx1 = x1;
      cy1 = y1 - curveOffset;
    } else {
      cx1 = x1;
      cy1 = y1 + curveOffset;
    }
    
    if (toPortArg.side === 'right') {
      cx2 = x2 + curveOffset;
      cy2 = y2;
    } else if (toPortArg.side === 'left') {
      cx2 = x2 - curveOffset;
      cy2 = y2;
    } else if (toPortArg.side === 'top') {
      cx2 = x2;
      cy2 = y2 - curveOffset;
    } else {
      cx2 = x2;
      cy2 = y2 + curveOffset;
    }
    
    return {
      path: `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`,
      midPoint: {
        x: (x1 + x2) / 2 + (fromPortArg.side === 'right' ? 20 : fromPortArg.side === 'left' ? -20 : 0),
        y: (y1 + y2) / 2 + (fromPortArg.side === 'bottom' ? 20 : fromPortArg.side === 'top' ? -20 : 0)
      }
    };
  };

  // Call useMemo unconditionally
  const pathInfo = useMemo(() => {
    return createPath(fromPort, toPort);
  }, [fromPort, toPort]);
  
  const { path, midPoint } = pathInfo;

  // Now do the early return after all hooks have been called
  if (!fromNode || !toNode || !fromPort || !toPort) {
    return null; // Skip rendering if we don't have the required data
  }

  // Determine arrow style and class
  const svgArrowClass = type === 'information' ? 'svg-arrow information-flow' : 'svg-arrow material-flow';
  const arrowColor = type === 'information' ? '#3498db' : '#e74c3c';
  const strokeStyle = type === 'information' ? 'dashed' : 'solid';
  const strokeWidth = type === 'information' ? 2 : 3;
  
  // Higher z-index for production-related arrows
  const zIndex = flow.from.includes('production') || flow.to.includes('production') ? 11 : 5;

  return (
    <div className="svg-arrow-container" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex }}>
      <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
        <defs>
          <marker
            id={`arrowhead-${flow.id}`}
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill={arrowColor} />
          </marker>
        </defs>
        <path
          d={path}
          stroke={arrowColor}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeStyle === 'dashed' ? '5,5' : 'none'}
          fill="none"
          markerEnd={`url(#arrowhead-${flow.id})`}
          style={{ pointerEvents: 'auto', cursor: 'pointer' }}
          onClick={onClick}
        />
      </svg>
      <div
        className="flow-label"
        style={{
          position: 'absolute',
          left: `${midPoint.x}px`,
          top: `${midPoint.y}px`,
          transform: 'translate(-50%, -50%)',
          zIndex: zIndex + 1,
          pointerEvents: 'auto'
        }}
        onClick={onClick}
      >
        <span>{flow.frequency}</span>
      </div>
    </div>
  );
};

export default SVGArrow;