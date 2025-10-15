import { Module } from "@nestjs/common";
import { MongooseModule } from "@nestjs/mongoose";
import { ImagesController } from "./images/images.controller";
import { ImagesService } from "./images/images.service";
import { Image, ImageSchema } from "./images/images.schema";

@Module({
  imports: [
    MongooseModule.forRoot("mongodb+srv://manh:200406@npqm.5lvoaxo.mongodb.net/highway?retryWrites=true&w=majority"),
    MongooseModule.forFeature([{ name: Image.name, schema: ImageSchema }]),
  ],
  controllers: [ImagesController],
  providers: [ImagesService],
})
export class AppModule {}
