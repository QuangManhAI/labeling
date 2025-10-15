import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";
import { Document } from "mongoose";

@Schema()
export class Image extends Document {
    @Prop({required: true})
    fileName: string; // tên file

    @Prop({required: true})
    filePath: string; // đường dẫn file

    @Prop({default:Date.now})
    updated_at: Date; //thời gian cập nhật

    @Prop({type: [{
        label: {type: String, required: true},
        bbox: {type: [Number], required: true},
        confidence: {type: Number, required: true},
    }], default: []})
    annotations: {
        label: string;
        bbox: number[];
        confidence: number;
    }[];
}
 
export const ImageSchema = SchemaFactory.createForClass(Image);