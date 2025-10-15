import { Body, Controller, Get, Param, Post } from "@nestjs/common";
import { ImagesService } from "./images.service";
import * as path from "path";
import * as fs from 'fs';

@Controller("images")
export class ImagesController {
  constructor(private readonly service: ImagesService) {}

  @Get("list-files")
  async listFiles() {
    const dir = path.join(process.cwd(), "uploads/images");
    const files = fs.readdirSync(dir);
    const dbImages = await this.service.listAll();
    const dbSet = new Set(dbImages.map((img) => img.fileName));
    return files.map((f) => ({
      fileName: f,
      filePath: `/uploads/images/${f}`,
      hasBox: dbSet.has(f),
    }));
  }


  @Get(":fileName")
  async getByName(@Param("fileName") fileName: string) {
    return this.service.getByName(fileName);
  }

  @Post("infer")
  async infer(@Body("image_path") image_path: string) {
    const absPath = path.join(process.cwd(), image_path.replace(/^\/+/, ""));
    return this.service.getOrInferImage(absPath);
  }

  @Post("infer/model")
  async inferOnlyModel(@Body("image_path") imagePath: string) {
    const absPath = path.join(process.cwd(), imagePath.replace(/^\/+/, ""));
    return this.service.inferFromModel(absPath);
  }

  @Post("save")
  async save(@Body() body: any) {
    return this.service.saveImage({
      fileName: body.fileName,
      filePath: body.filePath,
      annotations: body.annotations,
    });
  }
}
