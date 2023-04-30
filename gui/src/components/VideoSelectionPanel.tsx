import React, { useState } from 'react';
import Video from './Video';

type VideoMeta = {
    name: string;
    fps: number;
    frames: number;
};

function VideoSelectionPanel() {

    const [videoMetas, setVideoMetas] = useState<VideoMeta[]>([]);
    async function onRefresh() {
        const response = await fetch("https://localhost:5432/list");
        const json: VideoMeta[] = await response.json();
        setVideoMetas(json);
    }

    function selectVideo(video: string) {
        
    }

    return (<div>
        <button onClick={onRefresh}>Refresh</button>
        {/* <button>Add</button> */}
        {videoMetas.map(v => <Video onClick={}></Video>)}
    </div>);
}

export default VideoSelectionPanel;