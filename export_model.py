import tensorflow as tf
import net

INPUT_SIZE=224
def export_saved_model(checkpoint_dir=None,export_dir=None):
    def _parse_image(example):
        image=tf.io.decode_jpeg(example,channels=3)
        image=tf.image.convert_image_dtype(image,tf.float32)
        image=tf.image.per_image_standardization(image)
        image=tf.image.resize_images(image,size=(INPUT_SIZE,INPUT_SIZE))
        return image

    images=tf.placeholder(dtype=tf.string,shape=(None,))
    images_handled=tf.map_fn(_parse_image,images,dtype=tf.float32)

    resnet_18=net.ResNet18()
    logits=resnet_18._build_net(images_handled,num_classes=5)
    score=tf.nn.softmax(logits)
    classfication=tf.argmax(score,axis=-1)
    checkpoint=tf.train.latest_checkpoint(checkpoint_dir)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        builder=tf.saved_model.Builder(export_dir)
        saver.restore(sess,checkpoint)
        x_tensor_info=tf.saved_model.build_tensor_info(images)
        score_tensor_info=tf.saved_model.build_tensor_info(score)
        classfication_tensor_info=tf.saved_model.build_tensor_info(classfication)
        
        signature=tf.saved_model.build_signature_def(
            inputs={"image_bytes":x_tensor_info},
            outputs={
                "score":score_tensor_info,
                "classfication":classfication_tensor_info
            },method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        
        builder.add_meta_graph_and_variables(sess=sess,tags=["serve"],signature_def_map={"flower_serving":signature})
        builder.save()

if __name__=="__main__":
    export_saved_model(checkpoint_dir="checkpoint",export_dir="saved_model/1")