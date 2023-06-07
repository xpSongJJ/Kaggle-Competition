# 注意事项
由于直接读取的训练文件太大，需要进行数据转换来降低内存消耗。以下是类型转换详情：  

| columns        | before  | after    |
|----------------|---------|----------|
| session_id     | int64   | int64    |
| index          | int64   | int16    |
| elapsed_time   | int64   | int32    |
| event_name     | object  | category |
| name           | object  | category |
| level          | int64   | int8     |
| page           | float64 | int8     |
| rooom_coor_x   | float64 | float16  |
| room_coor_y    | float64 | float16  |
| screen_coor_x  | float64 | float16  |
| screen_coor_y  | float64 | float16  |
| hover_duration | float64 | float32  |
| text           | object  | category |
| fqid           | object  | category |
| room_fqid      | object  | category |
| text_fqid      | object  | category |
| fullscreen     | float64 | None     |
| hq             | float64 | None     |
| music          | float64 | None     |
| level_group    | object  | category |
注：object可以看做是字符串类型