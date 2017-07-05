import numpy as np
import scipy.io
import glob
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential, Merge
from keras.utils.np_utils import to_categorical


def load_data(index_name):
    # load the data
    pose_data_all = []
    force_data_all = []
    next_action_label_all = []
    current_action_label_all = []

    mat_files = glob.glob('data/*.mat')
    for mat_file in mat_files:
        data = scipy.io.loadmat(mat_file)
        pose_num = data['pose_window_width'][0][0]
        force_num = data['force_window_width'][0][0]
        data = data['windows']

        pose_data = np.array(data[:, 3:3+pose_num])
        force_data = np.array(data[:, 3+pose_num:3+pose_num+force_num])
        next_action_label = np.array(data[:, -1]+1)
        current_action_label = np.repeat(index_name.index(mat_file.split('/')[-1][:-12]), pose_data.shape[0])

        print mat_file.split('/')[-1][:-12], pose_data.shape[0]

        pose_data_all.extend(pose_data)
        force_data_all.extend(force_data)
        next_action_label_all.extend(next_action_label)
        current_action_label_all.extend(current_action_label)

    pose_data_all = np.array(pose_data_all)
    force_data_all = np.array(force_data_all)
    next_action_label_all = np.array(next_action_label_all, dtype=np.uint8)
    current_action_label_all = np.array(current_action_label_all, dtype=np.uint8)

    # shuffle the data
    shuffle_index = np.arange(pose_data_all.shape[0])
    np.random.shuffle(shuffle_index)
    pose_data_all = pose_data_all[shuffle_index, :]
    force_data_all = force_data_all[shuffle_index, :]
    next_action_label_all = next_action_label_all[shuffle_index]
    next_action_label_vec = to_categorical(next_action_label_all, len(index_name))
    current_action_label_all = current_action_label_all[shuffle_index]
    current_action_label_vec = to_categorical(current_action_label_all, len(index_name))

    # split the data
    num_training = int(pose_data_all.shape[0] * 0.8)
    training_data = np.hstack((pose_data_all[:num_training, :], force_data_all[:num_training, :]))
    training_current_action = current_action_label_vec[:num_training, :]
    training_next_action = next_action_label_vec[:num_training, :]
    testing_data = np.hstack((pose_data_all[num_training:, :], force_data_all[num_training:, :]))
    testing_current_action = current_action_label_vec[num_training:, :]
    testing_next_action = next_action_label_vec[num_training:, :]
    return training_data, training_current_action, training_next_action, testing_data, testing_current_action, testing_next_action


def main():
    index_name = ['end', 'ignore', 'approach', 'move', 'grasp_left', 'grasp_right', 'ungrasp_left', 'ungrasp_right',
                  'twist', 'push', 'neutral', 'pull', 'pinch', 'unpinch']

    training_data, training_current_action, training_next_action, testing_data, testing_current_action, testing_next_action = load_data(index_name)

    # model = Sequential()
    # model.add(Dense(64, input_dim=training_data.shape[1]+len(index_name), activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(len(index_name), activation='softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adadelta',
    #               metrics=['accuracy'])
    #
    # model.fit(np.hstack((training_data, training_current_action)),
    #           training_next_action,
    #           nb_epoch=500,
    #           validation_split=0.2,
    #           batch_size=16)  # starts training
    #
    # model.save('model1.h5')
    # score = model.fit(np.hstack((testing_data, testing_current_action)), testing_next_action, batch_size=16)
    # print score

    left_branch = Sequential()
    left_branch.add(Dense(64, input_dim=training_data.shape[1], activation='relu'))
    left_branch.add(Dropout(0.3))
    left_branch.add(Dense(64, activation='relu'))
    left_branch.add(Dropout(0.3))
    left_branch.add(Dense(64, activation='relu'))
    left_branch.add(Dropout(0.3))
    left_branch.add(Dense(64, activation='relu'))

    right_branch = Sequential()
    right_branch.add(Dense(8, input_dim=len(index_name), activation='relu'))

    merged = Merge([left_branch, right_branch], mode='concat')

    model = Sequential()
    model.add(merged)
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(14, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit([training_data, training_current_action],
              training_next_action,
              nb_epoch=20000,
              validation_split=0.2,
              batch_size=16)  # starts training
    model.save('model2.h5')
    score = model.evaluate([testing_data, testing_current_action], testing_next_action, batch_size=16)
    print score


if __name__ == '__main__':
    main()
