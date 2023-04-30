import videosReducer, {VideosState, add, VideoMeta} from './videosSlice';

describe('videos reducer', () => {
    const initialState: VideosState = {
        videos: [],
        status: 'idle',
    };

    it('should handle initial state', () => {
        expect(videosReducer(undefined, {type: 'unknown'})).toEqual({
            videos: [],
            status: 'idle',
        })
    })

    it('should handle add', () => {
        const payload: VideoMeta = {
            name: 'n',
            width: 1,
            height: 2,
            fps: 3,
            frames: 4,
            thumbnail: 't',
        }
        const actual = videosReducer(initialState, add(payload));
        expect(actual.videos).toEqual([payload]);
    })
})