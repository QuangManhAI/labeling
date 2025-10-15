import { HttpException, HttpStatus, Injectable } from "@nestjs/common";
import { InjectModel } from "@nestjs/mongoose";
import { Model } from "mongoose";
import { Image } from "./images.schema";
import * as path from 'path';
import axios from "axios";

@Injectable()
export class ImagesService{
  constructor(@InjectModel(Image.name) private readonly model: Model<Image>){}

  private toPublicPath(absPath: string) {
    const base = "/uploads/images/";
    return absPath.includes(base) ? absPath : base + path.basename(absPath);
  }

  async inferFromModel(absPath: string) {
    const fileName = path.basename(absPath);
    const publicPath = this.toPublicPath(absPath);

    const { data } = await axios.post("http://localhost:9100/inference", {
      image_path: absPath,
    });

    return {
      fileName,
      filePath: publicPath,
      annotations: data.annotations,
      source: "infer",
    };
  }

  async getOrInferImage(absPath: string) {
    const fileName = path.basename(absPath);

    const exist = await this.model.findOne({ fileName }).lean();
    if (exist && exist.annotations?.length) {
      return { ...exist, source: "db" };
    }

    return this.inferFromModel(absPath);
  }

  async saveImage(data: Partial<Image>) {
    const exist = await this.model.findOne({fileName: data.fileName})
    if (exist) {
      return this.model.findByIdAndUpdate(
        exist._id,
        {...data, updated_at: new Date()},
        {new: true}
      );
    }
    data.filePath = this.toPublicPath(data.filePath ?? "" );
    return new this.model(data).save();
  }

  async listAll() {
    const images = await this.model.find({}, "fileName filePath updated_at").lean();
    return images;
  }

  async getByName(fileName: string) {
    const img = await this.model.findOne({ fileName }).lean();
    if (!img) throw new HttpException("Not found", HttpStatus.NOT_FOUND);
    return img;
  }
}