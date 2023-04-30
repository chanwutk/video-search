import { createAsyncThunk, createSlice, PayloadAction } from "@reduxjs/toolkit";
import { RootState } from "../../app/store";

export type VideoMeta = {
    name: string;
    width: number;
    height: number;
    fps: number;
    frames: number;
    thumbnail: string;
}

export type VideosState = {
    videos: VideoMeta[];
    status: 'idle' | 'loading' | 'failed';
};

export const initialState: VideosState = {
    videos: [],
    status: 'idle',
};

export const refreshAsync = createAsyncThunk(
    'videos/refresh',
    async () => {
        const response = await fetch('https://localhost:5432/list-videos');
        const videos: VideoMeta[] = await response.json();
        return videos;
    }
)

export const videosSlice = createSlice({
    name: 'videos',
    initialState,
    reducers: {
        add: (state, action: PayloadAction<VideoMeta>) => {
            state.videos.push(action.payload);
        },
    },
    extraReducers: builder => {
        builder
            .addCase(refreshAsync.pending, state => {
                state.status = 'loading';
            })
            .addCase(refreshAsync.fulfilled, (state, action) => {
                state.status = 'idle';
                state.videos = action.payload;
            })
            .addCase(refreshAsync.rejected, state => {
                state.status = 'failed';
            });
    }
});

export const {add} = videosSlice.actions;

export const selectVideos = (state: RootState) => state.videos;

export default videosSlice.reducer;