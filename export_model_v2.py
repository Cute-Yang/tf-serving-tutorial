from config import DECODE_MAP,IMAGE_SIZE
import tensorflow as tf
import net

classification=list(DECODE_MAP.values())


def _parse_func(example):
    def _decode_jpeg():
        return tf.image.decode_jpeg(example,channels=3)
    
    def _decode_png():
        return tf.image.decode_png(example,channels=3)
    
    flag=tf.image.is_jpeg(example)
    image=tf.cond(
        flag,true_fn=_decode_jpeg,false_fn=_decode_png
    )
    image=tf.image.convert_image_dtype(image,dtype=tf.float32)
    image=tf.image.per_image_standardization(image)
    image_resized=tf.image.resize_images(image,size=(IMAGE_SIZE,IMAGE_SIZE))
    return image_resized

def export_saved_model_v2(checkpoint_dir,saved_model_dir):
    checkpoint=tf.train.latest_checkpoint(checkpoint_dir)
    image_train=tf.placeholder(dtype=tf.string,shape=(None,),name="input_tensor")
    image_handled=tf.map_fn(_parse_func,image_train,dtype=tf.float32)
    builder=tf.saved_model.Builder(saved_model_dir)
    flower_classify=tf.constant(classification,dtype=tf.string)
    resnet_18=net.ResNet18()
    logits=resnet_18._build_net(image_handled,num_classes=5)
    score=tf.nn.softmax(logits)
    classify_index=tf.argmax(score,axis=-1)
    classify_index=tf.cast(classify_index,tf.int32)
    classify=tf.gather(flower_classify,indices=classify_index)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,checkpoint)
        input_tensor_info=tf.saved_model.build_tensor_info(image_train)
        score_info=tf.saved_model.build_tensor_info(score)
        classify_info=tf.saved_model.build_tensor_info(classify)
        signature=tf.saved_model.build_signature_def(
            inputs={"image_bytes":input_tensor_info},
            outputs={
                "score":score_info,
                "classify":classify_info
            },method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder.add_meta_graph_and_variables(sess=sess,tags=["serve"],signature_def_map={"flower_serving":signature})
        builder.save()

if __name__=="__main__":
    checkpoint_dir="checkpoint"
    saved_model_dir="saved_model_v2/1"
    export_saved_model_v2(checkpoint_dir=checkpoint_dir,saved_model_dir=saved_model_dir)